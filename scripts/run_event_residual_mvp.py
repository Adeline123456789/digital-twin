"""Simulation-only MVP entry. Stop at the first failed prerequisite."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learn import DATA, ROOT, SIM, digest, finish, record, save
from episode_format import source_for
from robot_agent.nominal_reference import (PlannerSpy, PrivilegedAccessAudit, ReferencePlayer,
                                           content_hash, freeze_payload)

RESULTS = ROOT / 'experiments/event_residual_mvp'


def freeze(report_path):
    report = json.loads(report_path.read_text())
    if not (report['task_success'] and report['recording_verified'] and report['active_arms'] == ['left', 'right']):
        raise ValueError('Native complete dual-model Episode must pass before freeze')
    source_for(report)
    ident = report['experiment_id']
    raw = DATA / 'self_evolution' / ident
    episode = DATA / 'complete_sim/succeed' / ident
    from episode_format import validate
    validation = validate(episode, raw, expected_success=True)
    rows = [json.loads(line) for line in (raw / 'trajectory.jsonl').read_text().splitlines()]
    identity = dict(reference_episode_id=ident,
                    source_hash=report['program_version']['source_sha256']['scripts/try_plug_pick_sim.py'],
                    task_config_hash=digest(raw / 'task.yaml'),
                    policy_hash=content_hash([report['program_version']['adapter_sha256'], report['actions']]),
                    initial_state_hash=digest(raw / 'initial_state.npz'))
    payload = freeze_payload(rows, identity)
    reference_hash = content_hash(payload)
    player = ReferencePlayer(payload, reference_hash)
    destination = DATA / 'self_evolution/event_residual_mvp' / ident / 'reference.json'
    if destination.exists():
        raise FileExistsError('Reference already frozen; use the existing manifest')
    save(destination, payload)
    manifest = dict(**identity, reference_hash=reference_hash, nominal_action_hash=player.nominal_action_hash,
                    dataset_relative_path=str(destination.relative_to(DATA)), sha256=digest(destination),
                    source_report=str(report_path.relative_to(ROOT)), source_report_hash=digest(report_path),
                    frames=len(player.frames), validation=validation,
                    evidence=[dict(dataset_relative_path=str(p.relative_to(DATA)), sha256=digest(p))
                              for p in (raw / 'trajectory.jsonl', raw / 'task.yaml', raw / 'initial_state.npz',
                                        episode / 'outcome.json', episode / 'proprio_stats/model.xml')])
    save(RESULTS / 'nominal_manifest.json', manifest)
    return manifest


def replay(manifest_path):
    import mujoco
    import numpy as np
    import yaml
    sys.path.insert(0, str(SIM / 'scripts'))
    import try_plug_pick_sim as native
    from piper_sim.config import load
    from piper_sim.mixed_recording import MixedEpisodeWriter, configure_rigid_collection, checked_physics_step
    from piper_sim.task_archive import archive_episode
    from piper_sim.backends.arm_layout import save_recorded_model
    from collect_plug_contact_probes import state_row
    from collect_plug_sim_episode import PHASES
    from contact_evidence import integration_snapshot

    manifest = json.loads(manifest_path.read_text())
    source_report = ROOT / manifest['source_report']
    if digest(source_report) != manifest['source_report_hash']:
        raise ValueError('Source report changed after freeze')
    source_record = json.loads(source_report.read_text())
    source_for(source_record)
    for evidence in manifest['evidence']:
        if digest(DATA / evidence['dataset_relative_path']) != evidence['sha256']:
            raise ValueError('Frozen evidence changed')
    reference_path = DATA / manifest['dataset_relative_path']
    if digest(reference_path) != manifest['sha256']:
        raise ValueError('Reference bytes changed')
    player = ReferencePlayer(json.loads(reference_path.read_text()), manifest['reference_hash'])
    source = DATA / 'self_evolution' / manifest['reference_episode_id']
    task = yaml.safe_load((source / 'task.yaml').read_text())
    out, result = record('event_residual_mvp/runs', 'Frozen nominal physical replay', 'simulation')
    ident = result['experiment_id']
    raw = DATA / 'self_evolution' / ident
    raw.mkdir()
    save(raw / 'reference_manifest.json', manifest)
    (raw / 'task.yaml').write_text(yaml.safe_dump(task))
    stage = DATA / 'self_evolution' / ('stage3_recording_' + ident)
    session = 'learning_' + ident
    sim = native.MujocoBackend(dict(load(), task_scene=task), cameras=True)
    profile = configure_rigid_collection(sim)
    profile['robot_gravity_compensation'] = True
    with np.load(source / 'initial_state.npz') as initial:
        for name in ('qpos', 'qvel', 'ctrl'):
            getattr(sim.data, name)[:] = initial[name]
    mujoco.mj_forward(sim.model, sim.data)
    np.savez(raw / 'initial_state.npz', qpos=sim.data.qpos, qvel=sim.data.qvel, ctrl=sim.data.ctrl)
    writer = MixedEpisodeWriter(stage, session, '000001', 'sim', 'yellow_pick_insert', metadata=dict(
        source_trial=str(raw), force_sensor_present=False, physics=profile, task_config=task,
        active_arms=list(sim.arms), state_feature_names=list(sim.layout.names),
        camera_channels=['cam_external', 'cam_left_wrist', 'cam_right_wrist'],
        camera_provenance='simulated RGB; no real calibration claim', ee_pose_frame='each arm base',
        collection_mode='frozen_reference_native_physics', reference_hash=manifest['reference_hash'],
        privileged_information='evaluator and recording only',
        success_definition='unchanged native stable seating plus complete frozen replay'))
    planner, audit = PlannerSpy(), PrivilegedAccessAudit()
    history, actual_actions = [], []
    worst, error, origin, start = 0., None, time.time(), time.monotonic()
    try:
        with patch.object(native.PoseIK, 'solve', planner.forbidden), patch.object(native, 'main', planner.forbidden):
            for index in range(len(player.frames)):
                frame = player.command(index)
                action = np.asarray(frame.nominal_action)
                sim.apply(sim.layout.command(action, sim.sim_time, index))
                check = checked_physics_step(sim, .0002)
                worst = max(worst, check['max_penetration_m'])
                row, obs = state_row(sim, origin + sim.sim_time, action, True)
                row.update({'valid/collision': check['valid'], 'contact/max_penetration_m': check['max_penetration_m'],
                            'phase_index': PHASES.index(frame.phase)})
                writer.append(row, {f'cam_{camera}': obs.images[key] for camera, key in
                                   [('external', 'overhead'), ('left_wrist', 'left_wrist'), ('right_wrist', 'right_wrist')]},
                              privileged=dict(phase=frame.phase, objects=obs.extras['objects'],
                                              task_metrics=obs.extras['task']['metrics'], contacts=check['contacts'],
                                              cameras=obs.extras['cameras'], appearance=obs.extras.get('appearance'),
                                              qvel=sim.data.qvel.tolist(), physics_state=integration_snapshot(sim.model, sim.data)))
                history.append(dict(time=sim.sim_time, phase=frame.phase, q=obs.position.tolist(), action=action.tolist(),
                                    plug=sim.data.body('task/yellow_plug').xpos.tolist(),
                                    plug_quaternion_wxyz=sim.data.body('task/yellow_plug').xquat.tolist(),
                                    grip=sim.data.body('right/gripper_center').xpos.tolist(), penetration=check['max_penetration_m']))
                actual_actions.append(sim.control_targets().tolist())
                if not check['valid']:
                    error = 'Penetration exceeds unchanged collection tolerance'
                    break
                if abs(sim.sim_time - frame.timestep) > 1e-8:
                    error = 'Frozen reference clock mismatch'
                    break
                if index % 150 == 0:
                    print('frozen replay', index + 1, '/', len(player.frames), frame.phase, flush=True)
        complete = len(history) == len(player.frames) and error is None
        success = bool(complete and sim.read().extras['task']['metrics']['success'])
        native_report = dict(profile=profile, complete_pick_insert=success, frames=len(history),
                             error=error or (None if success else 'Frozen reference failed native stable seating'),
                             last_phase=history[-1]['phase'], max_penetration_m=worst, hardware_commands=0,
                             initial_robot_pose='frozen native initial state', speed_deg_s=12.,
                             reference_hash=manifest['reference_hash'])
        save(raw / 'report.json', native_report)
        (raw / 'trajectory.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
        np.savez(raw / 'final_state.npz', qpos=sim.data.qpos, qvel=sim.data.qvel, ctrl=sim.data.ctrl)
        save_recorded_model(sim.model, writer.path / 'model.xml')
        writer.meta.update(source_trial_report=native_report)
        writer.close(success=success, policy_training_eligible=False, complete_pick_insert=success,
                     all_substeps_valid=error is None, max_penetration_m=worst, phase_names=PHASES)
        archive_episode(stage, session, '000001')
    finally:
        sim.close()
    result.update(task_success=success, result='success' if success else 'failure',
                  validation_status='SIMULATION_VERIFIED' if success else 'SIMULATION_FAILED',
                  active_arms=['left', 'right'], actions=[dict(profile='original', policy='frozen_reference',
                  reference_hash=player.reference_hash)], initial_state=dict(target_index=task['target_index'],
                  task_config_sha256=digest(raw / 'task.yaml')), observed_effects=[native_report],
                  execution_time=time.monotonic() - start, reference_hash=player.reference_hash,
                  nominal_action_hash=player.nominal_action_hash, executed_action_hash=content_hash(actual_actions),
                  planner_calls=planner.calls, runtime_llm_calls=0, legacy_skill_invocations=0,
                  twin_modifications=0, hardware_executions=0, **audit.record())
    result['evidence'] = [dict(dataset_relative_path=str(p.relative_to(DATA)), sha256=digest(p))
                          for p in raw.iterdir() if p.is_file()]
    result['lessons_learned'] = ['Physical replay; evaluator privileged fields are excluded from ReferencePlayer.']
    finish(out, result)
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/finalize_recordings.py'), ident,
                                '--stage', str(stage.relative_to(DATA))], cwd=ROOT, text=True,
                               capture_output=True, check=True)
    validation_dir = ROOT / completed.stdout.strip().splitlines()[-1].rsplit(' ', 1)[0]
    validation = json.loads((validation_dir / 'result.json').read_text())
    result['recording_report'] = str(validation_dir.relative_to(ROOT))
    result['recording_verified'] = validation['result'] == 'success' and validation['task_success'] == success
    reference_rows = [json.loads(line) for line in (source / 'trajectory.jsonl').read_text().splitlines()]
    result['max_state_error_vs_generator'] = max(float(np.max(abs(np.array(a['q']) - b['q'])))
                                                for a, b in zip(history, reference_rows))
    result['decision'] = 'PASS' if (success and result['recording_verified'] and
        result['nominal_action_hash'] == result['executed_action_hash'] and planner.calls == audit.policy_reads == 0) else 'NO-GO'
    result['failure_category'] = None if result['decision'] == 'PASS' else 'nominal interface failure'
    finish(out, result)
    save(RESULTS / 'stage1_replay.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze', 'replay'], required=True)
    parser.add_argument('--source-report', type=Path)
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = 'egl'
    if args.stage == 'freeze':
        if args.source_report is None:
            parser.error('--source-report is required for freeze')
        freeze(args.source_report.resolve())
    else:
        result = replay(RESULTS / 'nominal_manifest.json')
        if result['decision'] != 'PASS':
            raise SystemExit(2)


if __name__ == '__main__':
    main()

"""Record the existing offline trial once, using its native writer and physics checks."""
import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import yaml

from learn import SIM

sys.path.insert(0, str(SIM / 'scripts'))
import try_plug_pick_sim as trial_program
from collect_plug_contact_probes import state_row
from collect_plug_sim_episode import PHASES
from piper_sim.mixed_recording import MixedEpisodeWriter
from piper_sim.task_archive import archive_episode


def collect(trial, stage_root, session, arguments):
    backend = trial_program.MujocoBackend
    physics_step = trial_program.checked_physics_step
    origin = time.time()
    writer = None
    checks = []

    class RecordingBackend(backend):
        def __init__(self, cfg):
            super().__init__(cfg, cameras=True)

        def close(self):
            try:
                if writer is None:
                    return
                report = json.loads((trial / 'report.json').read_text())
                history = [json.loads(line) for line in (trial / 'trajectory.jsonl').read_text().splitlines()]
                assert len(writer.rows) == len(history), 'Recording must cover every physics interval'
                error = 0.
                for row, reference in zip(writer.rows, history):
                    row['phase_index'] = np.asarray(PHASES.index(reference['phase']))
                    for side, start in [('left', 0), ('right', 7)]:
                        q = np.r_[np.deg2rad(row[f'state/{side}_arm/joint_positions']),
                                  row[f'state/{side}_arm/gripper_width_m']]
                        error = max(error, float(np.max(abs(q - reference['q'][start:start + 7]))))
                assert error <= 1e-5, 'Recording changed the observed state'
                task = yaml.safe_load((trial / 'task.yaml').read_text())
                camera = task.get('camera_calibration')
                writer.meta.update(
                    source_trial_report=report, physics=report['profile'], task_config=task,
                    initial_robot_pose=report['initial_robot_pose'],
                    simulation_joint_speed_deg_s=report['speed_deg_s'],
                    camera_calibration=json.loads((SIM / camera).read_text()) if camera else 'nominal camera rig',
                    fixture_registration=task.get('alignment_status', 'nominal layout, not calibrated physical ground truth'))
                valid = all(check['valid'] for check in checks)
                success = bool(report['complete_pick_insert'] and valid and
                               self.read().extras['task']['metrics']['success'])
                mujoco.mj_saveLastXML(str(writer.path / 'model.xml'), self.model)
                writer.close(success=success, policy_training_eligible=success,
                             complete_pick_insert=success, all_substeps_valid=valid,
                             max_penetration_m=max(check['max_penetration_m'] for check in checks),
                             same_run_max_state_error=error, phase_names=PHASES)
                # Phase labels come from the unmodified program's completed trace.
                path = writer.path / 'privileged.jsonl'
                privileged = [json.loads(line) for line in path.read_text().splitlines()]
                for row, reference in zip(privileged, history):
                    row['phase'] = reference['phase']
                path.write_text(''.join(json.dumps(row) + '\n' for row in privileged))
                print('ARCHIVED', archive_episode(stage_root, session, '000001'), flush=True)
            finally:
                super().close()

    def record_step(sim, tolerance):
        nonlocal writer
        check = physics_step(sim, tolerance)
        checks.append(check)
        if writer is None:
            writer = MixedEpisodeWriter(stage_root, session, '000001', 'sim', 'yellow_pick_insert',
                metadata=dict(source_trial=str(trial), force_sensor_present=False,
                              ee_pose_frame='each arm base', privileged_information='simulator only',
                              collection_mode='inline_native_physics',
                              success_definition='original task program and native stable seating detector'))
        row, obs = state_row(sim, origin + sim.sim_time, sim.control_targets(), True)
        row['valid/collision'] = check['valid']
        row['contact/max_penetration_m'] = check['max_penetration_m']
        row['phase_index'] = -1
        writer.append(row, {name: obs.images[key] for name, key in
            [('cam_external', 'overhead'), ('cam_left_wrist', 'left_wrist'), ('cam_right_wrist', 'right_wrist')]},
            privileged=dict(objects=obs.extras['objects'], task_metrics=obs.extras['task']['metrics'],
                            contacts=check['contacts'], cameras=obs.extras['cameras'],
                            appearance=obs.extras.get('appearance'), qvel=sim.data.qvel.tolist()))
        return check

    # shortcut: this existing script has no observer hook; bind only its two imports
    # in this isolated invocation. Replace with a native hook if one becomes available.
    previous_argv = sys.argv
    trial_program.MujocoBackend = RecordingBackend
    trial_program.checked_physics_step = record_step
    sys.argv = [str(SIM / 'scripts/try_plug_pick_sim.py'), '--output', str(trial), *arguments]
    try:
        trial_program.main()
    finally:
        sys.argv = previous_argv
        trial_program.MujocoBackend = backend
        trial_program.checked_physics_step = physics_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--stage-root', type=Path, required=True)
    parser.add_argument('--session', required=True)
    args, remaining = parser.parse_known_args()
    collect(args.trial, args.stage_root, args.session, remaining)

"""One simulation prefix: video-inspired grasp, no expert joint data."""
import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ['MUJOCO_GL'] = 'egl'
from learn import SIM, DATA, ROOT, digest, finish, record, save
sys.path.insert(0, str(SIM / 'scripts'))
import try_plug_pick_sim as native
from scipy.spatial.transform import Rotation


def video_grasp(sim, original_init):
    original_init(sim)
    # Native main adds board Rz(90); cancel it for jaws along the holder row.
    sim.rotation = Rotation.from_euler('z', -90., degrees=True).as_matrix() @ sim.rotation


def set_opening(position):
    # Native generator owns this array; keep its recorded commands consistent.
    if len(position) != 14 or position[6] != .060 or position[13] != .060:
        raise ValueError('Expected native dual-arm generator initialization')
    position[13] = .040


def run(output, task, complete=False):
    original_state = native.MujocoBackend.set_observation_state
    original_ik = native.PoseIK.__init__
    state_calls = 0

    def state(sim, position):
        nonlocal state_calls
        state_calls += 1
        if state_calls == 2:  # reset, then native IK approach initialization
            set_opening(position)
        return original_state(sim, position)

    def ik(self, sim, side='right'):
        video_grasp(self, lambda instance: original_ik(instance, sim, side))

    arguments = ['--task', str(task), '--smooth-timing', '--speed', '12', '--grasp-height', '.020']
    if not complete:
        arguments.append('--stop-after-lift')
    with patch.object(native.MujocoBackend, 'set_observation_state', state), patch.object(native.PoseIK, '__init__', ik), patch.object(sys, 'argv', arguments):
        if complete:
            from collect_inline import collect
            collect(output, DATA / 'self_evolution' / ('stage3_recording_' + output.name),
                    'learning_' + output.name, arguments)
        else:
            sys.argv = [str(SIM / 'scripts/try_plug_pick_sim.py'), '--output', str(output), *arguments]
            native.main()
    if state_calls != 2:
        raise RuntimeError('Native initialization contract changed')


def complete_trial(task):
    import json
    import subprocess
    import time
    out, result = record('event_residual_mvp/grasp_candidates', 'Video posture / 40 mm complete simulation candidate', 'simulation')
    raw = DATA / 'self_evolution' / result['experiment_id']
    start = time.monotonic()
    run(raw, task, complete=True)
    native_report = json.loads((raw / 'report.json').read_text())
    success = bool(native_report['complete_pick_insert'] and native_report['max_penetration_m'] <= .0002)
    result.update(result='success' if success else 'failure', task_success=success,
                  validation_status='SIMULATION_VERIFIED' if success else 'SIMULATION_FAILED',
                  observed_effects=[native_report], execution_time=time.monotonic() - start,
                  active_arms=['left', 'right'], initial_state=dict(target_index=3, task_config_sha256=digest(raw / 'task.yaml')),
                  actions=[dict(profile='original', initial_right_opening_m=.040, grasp_board_relative_yaw_deg=0.,
                                adapter_sha256=digest(__file__), grasp_height_m=.020)],
                  runtime_llm_calls=0, legacy_skill_invocations=0, hardware_executions=0, twin_modifications=0,
                  evidence=[dict(dataset_relative_path=str(p.relative_to(DATA)), sha256=digest(p))
                            for p in raw.iterdir() if p.is_file()])
    finish(out, result)
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/finalize_recordings.py'), result['experiment_id'],
                               '--stage', 'self_evolution/stage3_recording_' + result['experiment_id']],
                               cwd=ROOT, capture_output=True, text=True, check=True)
    verification_dir = ROOT / completed.stdout.strip().splitlines()[-1].rsplit(' ', 1)[0]
    verified = json.loads((verification_dir / 'result.json').read_text())
    result['recording_report'] = str(verification_dir.relative_to(ROOT))
    result['recording_verified'] = verified['result'] == 'success' and verified['task_success'] == success
    result['decision'] = 'PASS' if success and result['recording_verified'] else 'NO-GO'
    finish(out, result)
    save(ROOT / 'experiments/event_residual_mvp/video_grasp_complete.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    if args.complete:
        complete_trial(args.task)
    else:
        if args.output is None:
            parser.error('--output required for prefix')
        run(args.output, args.task)

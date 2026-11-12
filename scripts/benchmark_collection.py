"""Matched native replay / inline collection experiment; raw evidence stays local."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import av
import h5py
import numpy as np

from learn import DATA, ROOT, SIM, digest, finish, save, simulate
from episode_format import source_for

ROUND = ROOT / 'experiments/research_stage3/round2'


def run_case(case, inline, label='Stage3 round2', kind='research_stage3'):
    out, result = simulate(kind, case['offset_mm'], case['shift_mm'],
                           label + ' ' + case['name'], record_inline=inline,
                           grasp_tilt_deg=case.get('grasp_tilt_deg'))
    ident = result['experiment_id']
    local = ROOT / '.local' / ident
    stage = DATA / 'self_evolution' / ('stage3_recording_' + ident)
    runtime = SIM / ('.venv-mujoco/bin/python' if os.environ.get('PIPER_SIM_PROFILE')=='retained_final' else '.venv-mujoco-hardware/bin/python')
    env = dict(os.environ, MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    replay_seconds = 0.
    verification_seconds = None
    result.update(privileged_information_used=True,
        privileged_fields_used=['joint/action state', 'object/goal pose', 'contact penetration',
                                'task success ground truth', 'camera pose', 'qvel'],
        tokens=0, planner_calls=0, research_orchestrator_tokens=None, human_interventions=None,
        collection_adapter_sha256=digest(ROOT / 'scripts/collect_inline.py'),
        comparison_case=case, llm_inference_seconds=0)
    try:
        if result['task_success'] is None:
            raise RuntimeError('Trial did not produce a physical result')
        if not inline:
            start = time.monotonic()
            command = [str(runtime), str(SIM / 'scripts/collect_plug_sim_episode.py'),
                       '--trial', str(DATA / 'self_evolution' / ident), '--root', str(stage),
                       '--session', 'learning_' + ident]
            with (local / 'recording.log').open('w') as log:
                subprocess.run(command, cwd=SIM, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1200)
            replay_seconds = time.monotonic() - start
        start = time.monotonic()
        completed = subprocess.run([str(runtime), str(ROOT / 'scripts/finalize_recordings.py'),
            ident, '--stage', str(stage.relative_to(DATA))], cwd=ROOT, env=env,
            text=True, capture_output=True, check=True, timeout=120)
        verification_seconds = time.monotonic() - start
        (local / 'verification.log').write_text(completed.stdout + completed.stderr)
        verification_dir = ROOT / completed.stdout.strip().splitlines()[-1].rsplit(' ', 1)[0]
        verification = json.loads((verification_dir / 'result.json').read_text())
        assert verification['result'] == 'success', verification.get('error')
        assert verification['task_success'] == result['task_success']
        source_for(result)
        result['recording_report'] = str(verification_dir.relative_to(ROOT))
        result['recording_verified'] = True
        result['source_scripts_src_configs_unchanged'] = True
    except Exception as exc:
        result['recording_verified'] = False
        result['recording_error'] = str(exc).replace(str(ROOT), '$PROJECT_ROOT').replace(str(DATA), '$DATA')
    if result['recording_verified']:
        from diagnose_retention import diagnose_episode
        try:
            result['causal_diagnosis'] = diagnose_episode(result)
        except Exception as exc:
            result['causal_diagnosis'] = {'status':'UNKNOWN','error_type':type(exc).__name__}
    result.update(replay_seconds=replay_seconds, verify_install_process_seconds=verification_seconds)
    result['collection_component_seconds'] = result['execution_time'] + replay_seconds + (verification_seconds or 0.)
    result['lessons_learned'] = ['Simulation-only; privileged fields disclosed in result.json; complete recording required for cost denominator.']
    result['next_improvement'] = 'Compare complete paired trajectories and recordings before promotion'
    finish(out, result)
    paths = [str(out.relative_to(ROOT))]
    if result.get('recording_report'):
        paths.append(result['recording_report'])
    subprocess.run([sys.executable, str(ROOT / 'scripts/publish.py'), '--message',
                    'Record Stage3 collection comparison ' + ident, *paths], check=True)
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'HEAD'], cwd=ROOT, text=True).split()[0]
    assert remote == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return result


def compare_pair(a, b):
    errors = {}
    trials = [DATA / 'self_evolution' / r['experiment_id'] for r in (a, b)]
    trajectories = [[json.loads(line) for line in (t / 'trajectory.jsonl').read_text().splitlines()] for t in trials]
    assert len(trajectories[0]) == len(trajectories[1])
    assert [r['phase'] for r in trajectories[0]] == [r['phase'] for r in trajectories[1]]
    for key in ('time', 'q', 'action', 'plug', 'plug_quaternion_wxyz', 'grip', 'penetration'):
        errors[key] = float(np.max(abs(np.array([r[key] for r in trajectories[0]]) -
                                      np.array([r[key] for r in trajectories[1]]))))
    for name in ('initial_state.npz', 'final_state.npz'):
        with np.load(trials[0] / name) as x, np.load(trials[1] / name) as y:
            for key in x.files:
                errors[name + '/' + key] = float(np.max(abs(x[key] - y[key])))
    episodes = [next((DATA / 'self_evolution' / ('stage3_recording_' + r['experiment_id'])).glob('complete_sim/*/*')) for r in (a, b)]
    with h5py.File(episodes[0] / 'proprio_stats/proprio_stats.h5') as x, h5py.File(episodes[1] / 'proprio_stats/proprio_stats.h5') as y:
        def check(key, value):
            if isinstance(value, h5py.Dataset) and key not in ('timestamp', 'timestamp_monotonic'):
                errors['h5/' + key] = float(np.max(abs(value[:].astype(float) - y[key][:].astype(float))))
        x.visititems(check)
    pixels = {}
    mean_pixels = {}
    video_properties = {}
    for video in (episodes[0] / 'videos').glob('*.mp4'):
        with av.open(str(video)) as x, av.open(str(episodes[1] / 'videos' / video.name)) as y:
            sx, sy = x.streams.video[0], y.streams.video[0]
            video_properties[video.name] = ((sx.width, sx.height, sx.average_rate, sx.codec_context.name) ==
                                          (sy.width, sy.height, sy.average_rate, sy.codec_context.name))
            maximum = 0
            means = []
            for f, g in zip(x.decode(video=0), y.decode(video=0), strict=True):
                difference = abs(f.to_ndarray(format='rgb24').astype(np.int16) - g.to_ndarray(format='rgb24').astype(np.int16))
                maximum = max(maximum, int(difference.max()))
                means.append(float(difference.mean()))
            pixels[video.name] = maximum
            mean_pixels[video.name] = {'mean': float(np.mean(means)), 'max_frame_mean': max(means)}
    privileged = [[json.loads(line) for line in (episode / 'proprio_stats/privileged.jsonl').read_text().splitlines()] for episode in episodes]
    physical_equal = all(v <= 1e-10 for v in errors.values()) and a['task_success'] == b['task_success']
    return dict(max_numeric_errors=errors, max_decoded_pixel_errors=pixels,
                decoded_pixel_mae=mean_pixels, video_properties_equal=video_properties,
                privileged_trace_equal=privileged[0] == privileged[1], pixel_exact=not any(pixels.values()),
                physical_equivalent=physical_equal,
                equivalent=physical_equal and privileged[0] == privileged[1] and all(video_properties.values()),
                baseline_s=a['collection_component_seconds'], candidate_s=b['collection_component_seconds'],
                reduction_fraction=1-b['collection_component_seconds']/a['collection_component_seconds'])


def main():
    design = json.loads((ROUND / 'design.json').read_text())
    # First pair is frozen as a failed pixel-exact gate; do not rerun or relabel it.
    results = []
    for case in design['cases'][1:]:
        pair = {}
        for method in case['order']:
            result = run_case(case, method == 'inline')
            pair[method] = result
            save(ROUND / 'amended_progress.json', {'completed_pairs': results, 'current_ids': {k: v['experiment_id'] for k, v in pair.items()}})
            if not result['recording_verified']:
                raise RuntimeError('Recording rejected; preserve report and stop this comparison')
        comparison = compare_pair(pair['baseline'], pair['inline'])
        results.append(dict(case=case, trial_ids={k: v['experiment_id'] for k, v in pair.items()}, **comparison))
        save(ROUND / 'amended_comparison.json', results)
        print('PAIR', case['name'], json.dumps(comparison), flush=True)
        if not comparison['equivalent']:
            raise RuntimeError('Candidate changed physical trace, camera provenance or video properties; no promotion')


if __name__ == '__main__':
    main()

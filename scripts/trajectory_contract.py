"""Named simulation-command to existing Piper replay contract; no hardware imports."""
import hashlib
import json
from pathlib import Path
import numpy as np

RIGHT = [*[f'right.joint{i}' for i in range(1, 7)], 'right.gripper_width']

def right_targets(rows, names):
    if len(set(names)) != len(names):
        raise ValueError('Ambiguous feature names')
    actions = np.asarray([r['action'] for r in rows], dtype=float)
    if actions.ndim != 2 or actions.shape[1] != len(names):
        raise ValueError('Action shape does not match declared feature names')
    selected = actions[:, [names.index(n) for n in RIGHT]].copy()
    selected[:, :6] = np.rad2deg(selected[:, :6])
    selected[:, 6] /= .095
    if not np.isfinite(selected).all() or np.any((selected[:, 6] < 0) | (selected[:, 6] > 1)):
        raise ValueError('Invalid right reference commands')
    return selected

def resample(rows, names, max_joint_speed=12., hz=100):
    source = np.asarray([r['time'] for r in rows], dtype=float)
    source -= source[0]
    targets = right_targets(rows, names)
    if len(source) < 2 or not np.isfinite(source).all() or np.any(np.diff(source) <= 0):
        raise ValueError('Increasing finite source timestamps required')
    velocity = np.abs(np.diff(targets, axis=0) / np.diff(source)[:, None])
    if velocity[:, :6].max() > max_joint_speed+1e-6 or velocity[:, 6].max() > .5+1e-6:
        raise ValueError('Reference exceeds existing task speed; regenerate in simulation, do not silently stretch')
    timeline = np.r_[np.arange(0, source[-1], 1/hz), source[-1]]
    samples = np.column_stack([np.interp(timeline, source, targets[:, j]) for j in range(7)])
    phases = [rows[i]['phase'] for i in np.searchsorted(source, timeline, side='right')-1]
    return timeline, samples, phases

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze(episode, trial, output):
    episode, trial, output = map(Path, (episode, trial, output))
    meta = json.loads((episode/'proprio_stats/metadata.json').read_text())
    if meta['domain'] != 'sim' or meta['active_arms'] != ['right'] or not meta['success']:
        raise ValueError('A verified right-only successful simulation Episode is required')
    rows = [json.loads(s) for s in (trial/'trajectory.jsonl').read_text().splitlines()]
    timeline, targets, phases = resample(rows, meta['state_feature_names'])
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output/'reference.npz', time=timeline, targets=targets, phases=np.asarray(phases))
    manifest = dict(trajectory_id=sha(output/'reference.npz'), source_episode=episode.name,
        source_trajectory_sha256=sha(trial/'trajectory.jsonl'), source_initial_state_sha256=sha(trial/'initial_state.npz'),
        adapter_sha256=sha(__file__), feature_names=RIGHT, units=['degree']*6+['width_mm / 95'],
        command_provenance='simulation reference action, NOT simulated measured q',
        adaptation='linear interpolation on original relative timeline at100Hz; no smoothing or time stretching',
        source_time_origin_s=rows[0]['time'], duration_s=float(timeline[-1]), steps=len(timeline),
        first_target=targets[0].tolist(), last_target=targets[-1].tolist(),
        simulation_validation='PENDING adapted-reference physics replay', real_execution='NOT_EXECUTED')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def load_reference(directory, require_validation=False):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if sha(directory/'reference.npz') != manifest['trajectory_id']:
        raise ValueError('Frozen reference hash mismatch')
    if require_validation:
        validation=json.loads((directory/'validation.json').read_text())
        if (validation['reference_sha256'] != manifest['trajectory_id'] or
                not validation['task_success'] or not validation['all_substeps_valid']):
            raise ValueError('Adapted reference has not passed physical simulation')
    with np.load(directory/'reference.npz',allow_pickle=False) as data:
        return data['time'].copy(),data['targets'].copy(),data['phases'].tolist()

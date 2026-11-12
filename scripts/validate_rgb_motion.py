"""RGB-only finger motion audit; pixel displacement is not calibrated aperture."""
import csv
import json
import time
import cv2
import h5py
import numpy as np
from learn import DATA, ROOT, record, finish, digest, save
from observe_real_failures import camera_monotonic_times


def track_step(previous, current, points):
    forward, s1, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                            winSize=(25, 25), maxLevel=3)
    backward, s2, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None,
                                             winSize=(25, 25), maxLevel=3)
    return forward, s1.ravel().astype(bool) & s2.ravel().astype(bool), np.linalg.norm(backward-points, axis=1)


def relative_span(tracks, labels, survivors):
    # Fixed feature identities prevent changing medians from appearing as motion.
    displacement = tracks-tracks[0]
    left = np.median(displacement[:, survivors & (labels == 0), 0], axis=1)
    right = np.median(displacement[:, survivors & (labels == 1), 0], axis=1)
    return right-left


def slope(x, y):
    return float(x@y/(x@x))


def episode(name, design, local):
    ep = DATA/'complete_real/failed'/name
    video = ep/'videos/cam_right_wrist_rgb.mp4'
    h5 = ep/'proprio_stats/proprio_stats.h5'
    command = ep/'raw/task/commanded_probe.csv'
    rows = list(csv.DictReader(command.open()))
    ct = np.array([float(x['monotonic']) for x in rows])
    with h5py.File(h5) as f:
        timestamps = camera_monotonic_times(f, 'cam_right_wrist')
    phases = np.array([rows[i]['phase'] for i in np.clip(np.searchsorted(ct, timestamps, side='right')-1, 0, len(rows)-1)])
    selection = np.flatnonzero(np.isin(phases, ['grasp', 'lift']))
    start, end = int(selection[0]), int(selection[-1])
    cap = cv2.VideoCapture(str(video))
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(timestamps):
        cap.release(); raise ValueError('Video/HDF5 frame count mismatch')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, first = cap.read()
    if not ok:
        cap.release(); raise ValueError('Missing first RGB frame')
    previous = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    groups = []
    for x0, y0, x1, y1 in design['rois'].values():
        mask = np.zeros_like(previous); mask[y0:y1, x0:x1] = 255
        mask[previous >= 85] = 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        points = cv2.goodFeaturesToTrack(previous, 30, .01, 4, mask=mask)
        if points is None:
            cap.release(); raise ValueError('No visible features in a frozen ROI')
        groups.append(points.reshape(-1, 2))
    labels = np.concatenate([np.full(len(p), i) for i, p in enumerate(groups)])
    points = np.vstack(groups); alive = np.ones(len(points), bool)
    tracks = [points.copy()]; errors = []; snapshots = {start: first}
    try:
        for i in range(start+1, end+1):
            ok, image = cap.read()
            if not ok: raise ValueError('Unexpected end of RGB recording')
            current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            points, status, error = track_step(previous, current, points)
            alive &= status & (error <= design['criteria']['max_forward_backward_px'])
            tracks.append(points.copy()); errors.append(error); previous = current
            if phases[i] != phases[i-1] or i == end: snapshots[i] = image
    finally:
        cap.release()
    tracks = np.array(tracks); counts = [int(np.sum(alive & (labels == i))) for i in range(3)]
    fractions = [counts[i]/len(groups[i]) for i in range(3)]
    sufficient = all(counts[i] >= design['criteria']['min_features_each_finger'] and
                     fractions[i] >= design['criteria']['min_survivor_fraction'] for i in (0, 1))
    for i, image in snapshots.items():
        for p, label, kept in zip(tracks[i-start], labels, alive):
            cv2.circle(image, tuple(np.rint(p).astype(int)), 3,
                       ((0, 255, 0), (255, 0, 0), (0, 255, 255))[label] if kept else (0, 0, 255), 1)
        cv2.imwrite(str(local/f'{name}_{i}.jpg'), image)
    ts = timestamps[start:end+1]; phase = phases[start:end+1]
    channels = {key: np.interp(ts, ct, [float(r[key]) for r in rows])
                for key in ('feedback_gripper_open', 'target_gripper_open')}
    for key in channels: channels[key] -= channels[key][0]
    fixture = np.median(tracks[:, alive & (labels == 2)]-tracks[0, alive & (labels == 2)], axis=1) if counts[2] else None
    fixture_motion = float(np.linalg.norm(fixture[phase == 'grasp'], axis=1).max()) if fixture is not None else None
    geometry_ok = sufficient and fixture_motion is not None and fixture_motion <= design['criteria']['max_grasp_fixture_displacement_px']
    result = dict(source_episode=name, first_frame=start, last_frame=end, initial_features=[len(g) for g in groups],
                  surviving_features=counts, survivor_fractions=fractions, tracking_usable=bool(sufficient),
                  grasp_fixture_max_motion_px=fixture_motion, geometry_audit_pass=bool(geometry_ok),
                  evidence=[dict(dataset_relative_path=str(p.relative_to(DATA)), sha256=digest(p)) for p in (video, h5, command)])
    series = dict(frame=list(range(start, end+1)), time_s=(ts-ts[0]).tolist(), phase=phase.tolist(),
                  **{k:v.tolist() for k,v in channels.items()},
                  relative_finger_span_px=relative_span(tracks, labels, alive).tolist() if sufficient else None)
    save(local/f'{name}_tracks.json', dict(tracks=tracks.tolist(), labels=labels.tolist(), survivors=alive.tolist()))
    return result, series


def main():
    out, r = record('research_stage2', 'RGB-only visible finger response validation', 'real_historical_rgb_only')
    started = time.monotonic(); local = DATA/'calibration'/r['experiment_id']; local.mkdir()
    design = json.loads((ROOT/'experiments/research_stage2/round6/design.json').read_text())
    summaries = []; series = {}
    try:
        for name in [design['train_episode']]+design['heldout_episodes']:
            summary, trace = episode(name, design, local); summaries.append(summary); series[name] = trace
        r['observed_effects'] = summaries
        if all(x['geometry_audit_pass'] for x in summaries):
            train = series[design['train_episode']]; fit = np.array(train['phase']) == 'grasp'
            y = np.array(train['relative_finger_span_px'])[fit]
            models = {key:slope(np.array(train[key])[fit], y) for key in ('feedback_gripper_open', 'target_gripper_open')}
            evaluations = []
            for name, trace in series.items():
                for phase in ('grasp', 'lift'):
                    sel = np.array(trace['phase']) == phase
                    rmse = {key:float(np.sqrt(np.mean((models[key]*np.array(trace[key])[sel]-np.array(trace['relative_finger_span_px'])[sel])**2))) for key in models}
                    evaluations.append(dict(source_episode=name, role='train' if name == design['train_episode'] else 'heldout', phase=phase, rmse_px=rmse))
            r['measurement_models'] = models; r['measurement_evaluation'] = evaluations
        r.update(result='success', task_success=None, validation_status='RGB_OBSERVABILITY_AUDIT_ONLY',
                 initial_state=design, actions=['Frozen-ROI forward/backward optical flow; no depth inputs'],
                 expected_effects=['Check visible finger motion against command and measured motor feedback'],
                 repair={'physics_changed':False, 'calibration_promoted':False},
                 root_cause='UNKNOWN: image tracking and motor feedback alone do not identify contact physics',
                 lessons_learned=['RGB pixel motion must not be reported as metric aperture or depth'],
                 next_improvement='Inspect tracking and independent-recording residuals; stop if measurement criteria fail')
        save(out/'frame_features.json', series)
        r['evidence'] = [e for summary in summaries for e in summary['evidence']]
    except Exception as exc:
        r['error'] = str(exc); r['observed_effects'] = summaries
    finally:
        r['execution_time'] = time.monotonic()-started; r['tokens'] = {'planner':0, 'research_orchestrator':None}
        r['analysis_version'] = digest(__file__); r['simulation_seconds'] = 0
        finish(out, r)


if __name__ == '__main__': main()

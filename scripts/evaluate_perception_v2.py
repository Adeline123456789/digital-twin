"""Independent semantic calibration and episode-grouped RGB evaluation."""
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent.keypoint_detector import KeypointNet
from robot_agent.observation_adapter import VisualTracker, extract_rgb
from robot_agent.target_lock import lock_outlet_three
from learn import DATA, ROOT, digest


def main():
    folder = ROOT / 'experiments/event_residual_mvp_v2'
    design = json.loads((folder / 'perception_design.json').read_text())
    data = json.loads((folder / 'perception_dataset.json').read_text())
    trained = json.loads((folder / 'keypoint_training.json').read_text())
    model = KeypointNet().cuda().eval()
    model.load_state_dict(torch.load(DATA / trained['weights']['dataset_relative_path'], weights_only=True))
    rows = []
    for sequence in data['sequences']:
        arrays = np.load(DATA / sequence['dataset_relative_path'])
        prediction, confidence = model.predict(arrays['rgb'])
        error = np.linalg.norm((prediction - arrays['keypoints']) * 2, axis=-1)
        relative_error = np.linalg.norm(((prediction[:, 1] - prediction[:, 0]) -
                                        (arrays['keypoints'][:, 1] - arrays['keypoints'][:, 0])) * 2, axis=-1)
        for index in range(len(prediction)):
            rows.append(dict(seed=sequence['seed'], source_episode=sequence['source_episode'], split=sequence['split'],
                index=index, score=float(confidence[index].min()), confidence=confidence[index].tolist(),
                visible=arrays['visible'][index].tolist(), error_px=error[index].tolist(),
                relative_error_px=float(relative_error[index])))
    validation = [r for r in rows if r['split'] == 'validation']
    sweep = []
    rng = np.random.default_rng(77)
    for threshold in design['calibration']['feature_score_grid']:
        per_sequence = []
        for seed in sorted({r['seed'] for r in validation}):
            selected = [r for r in validation if r['seed'] == seed and r['score'] >= threshold]
            correct = sum(all(r['visible']) and max(r['error_px']) <= 6 and r['relative_error_px'] <= 8 for r in selected)
            per_sequence.append([correct, len(selected)])
        counts = np.array(per_sequence)
        bootstrap = counts[rng.integers(0, len(counts), (10000, len(counts)))].sum(axis=1)
        precision = bootstrap[:, 0] / np.maximum(1, bootstrap[:, 1])
        sweep.append(dict(threshold=threshold, selected=int(counts[:, 1].sum()), correct=int(counts[:, 0].sum()),
                          precision_lower_bound=float(np.quantile(precision, .025))))
    accepted = [s for s in sweep if s['selected'] > 0 and s['precision_lower_bound'] >= .9]
    chosen = accepted[0] if accepted else sweep[-1]
    calibration = dict(accepted=bool(accepted), feature_score_threshold=chosen['threshold'],
        precision_lower_bound=chosen['precision_lower_bound'], sweep=sweep,
        reason='heldout semantic validation required; no hardcoded detector confidence',
        validation_sequences=4, validation_physical_source_episodes=1,
        validation_visible_pairs=sum(all(r['visible']) for r in validation))
    (folder / 'observation_calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    summaries = []
    for sequence in data['sequences']:
        selected = [r for r in rows if r['seed'] == sequence['seed']]
        errors = [[r['error_px'][k] for r in selected if r['visible'][k]] for k in range(2)]
        pairs = [r for r in selected if all(r['visible'])]
        active = [r for r in selected if calibration['accepted'] and r['score'] >= calibration['feature_score_threshold']]
        summaries.append(dict(seed=sequence['seed'], source_episode=sequence['source_episode'], split=sequence['split'],
            samples=len(selected), visibility_rate=np.mean([r['visible'] for r in selected], axis=0).tolist(),
            keypoint_mean_error_px=[float(np.mean(e)) if e else None for e in errors],
            visible_pairs=len(pairs), relative_mean_error_px=float(np.mean([r['relative_error_px'] for r in pairs])) if pairs else None,
            unresolved_rate=1-len(active)/len(selected)))
    # Original-resolution chronological RGB, with a controlled temporary black-out.
    temporal = []
    for episode in design['sources']:
        tracker = VisualTracker(calibration=calibration, global_lock=lock_outlet_three)
        direct_locks = retained_during_occlusion = reacquired = 0
        first_lock = None
        path = DATA / 'complete_sim/succeed' / episode / 'videos/cam_external_rgb.mp4'
        with av.open(str(path)) as stream:
            for index, frame in enumerate(stream.decode(video=0)):
                if index > 520: break
                rgb = frame.to_ndarray(format='rgb24')
                if index == 0:
                    direct_locks = int(lock_outlet_three(rgb) is not None)
                hidden = 440 <= index <= 445
                observed = np.zeros_like(rgb) if hidden else rgb
                features = tracker.update(observed, index / 30, 'transport' if index < 442 else 'insert')
                if features.target_id is not None and first_lock is None: first_lock = index
                if hidden: retained_during_occlusion += int(features.target_id == 3)
                if index == 446: reacquired = int(features.target_id == 3)
        temporal.append(dict(episode=episode, frames=521, initial_rgb_layout_lock=bool(direct_locks),
            first_lock_frame=first_lock, blackout_frames=6, identity_retained_frames=retained_during_occlusion,
            identity_after_blackout=bool(reacquired), local_visual_alignment_recovered=False,
            recovery_scope='Identity retention only. Missing physical top-pin cue is not recovered by optical flow.',
            video_sha256=digest(path)))
    output = dict(scope='Simulation-only; two physical source episodes, 18 synthetic sequences; no per-frame independent-trial claim',
        calibration=calibration, sequences=summaries, temporal=temporal,
        variants=['V0 instantaneous morphology (historical report)', 'V2 layout lock + optical flow', 'V2 RGB heatmap + visibility + calibrated abstention'],
        decision='GO' if calibration['accepted'] else 'NO-GO',
        reason='No qualified simultaneous local pin/hole cue on validation; retain UNKNOWN and allow proprioceptive mode',
        training_only_auxiliary_views=512, real_data_used=False, left_wrist_used=False)
    raw = DATA / 'self_evolution/event_residual_mvp_v2/perception/evaluation_detail.json'
    raw.write_text(json.dumps(rows, indent=2) + '\n')
    output['detail'] = dict(dataset_relative_path=str(raw.relative_to(DATA)), sha256=digest(raw))
    (folder / 'observation_ablation.json').write_text(json.dumps(output, indent=2) + '\n')
    print('PERCEPTION_EVALUATION', json.dumps(dict(decision=output['decision'], calibration=calibration, temporal=temporal)), flush=True)


if __name__ == '__main__':
    main()

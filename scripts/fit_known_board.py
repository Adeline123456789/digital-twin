"""Fit one rigid, metric RGB board directly in pixels using measured joint FK.

Reuse the existing FK-constrained projector (SHECAR's reprojection formulation).
Only the camera mount and one fixed board pose are variable. This is an offline
candidate: it does not write simulator or hardware configuration.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from chessboard_observation import board_points
from fit_probe_registration import project
from learn import digest, save


def displaced(pose, delta):
    result = pose.copy()
    result[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ pose[:3, :3]
    result[:3, 3] += delta[3:]
    return result


def run(source, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    paths = [source / p for p in (
        'fit_input_003.json', 'observations_002/private_target_observations.json',
        'observations_001/private_robot_observations.json',
        'episodes/20261116_real_board_probe_000002/raw/task/right_wrist/camera.json')]
    split, observed, robot, camera = [json.loads(p.read_text()) for p in paths]
    mount = np.array(split['baseline_T_gripper_camera'])
    poses = np.array(robot['T_base_gripper'])
    train_names = {p for p, s in zip(split['provenance']['phase'], split['split']) if s == 'train'}
    heldout_names = {p for p, s in zip(split['provenance']['phase'], split['split']) if s == 'heldout'}
    selected = [i for i, p in enumerate(robot['phase']) if p in train_names | heldout_names]
    records = []
    for i in selected:
        observation = observed[i]
        uv = cv2.undistortPoints(np.array(observation['corners_px']).reshape(-1, 1, 2),
            np.array(camera['K']), np.array(camera['distortion_coeffs']), P=np.array(camera['K'])).reshape(-1, 2)
        ids = observation.get('corner_ids', range(88))
        records.extend((i, j, *pixel) for j, pixel in zip(ids, uv))
    rows = np.array(records)
    vi, pi, uv = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2:]
    train = np.array([robot['phase'][i] in train_names for i in vi])
    closed = np.array(split['T_base_gripper']) @ mount @ np.array(split['T_camera_target'])
    used = np.array(split['split']) == 'train'
    board = np.eye(4)
    board[:3, :3] = Rotation.from_matrix(closed[used, :3, :3]).mean().as_matrix()
    board[:3, 3] = closed[used, :3, 3].mean(0)
    points, K = board_points(), np.array(camera['K'])

    def unpack(x, adjustable):
        m = displaced(mount, x[:6]) if adjustable else mount
        b = displaced(board, x[-6:])
        return m, b, points @ b[:3, :3].T + b[:3, 3]

    def optimize(adjustable):
        bound = np.r_[[np.deg2rad(15)] * 3, [.05] * 3,
                      [np.deg2rad(20)] * 3, [.08] * 3]
        if not adjustable:
            bound = bound[6:]
        def residual(x):
            m, _, p = unpack(x, adjustable)
            return (project(p, poses, m, K, vi[train], pi[train])[0] - uv[train]).ravel()
        return least_squares(residual, np.zeros(len(bound)), bounds=(-bound, bound),
                             x_scale='jac', loss='soft_l1', f_scale=1, max_nfev=200)

    baseline = optimize(False)
    candidate = optimize(True)

    def metrics(fit, adjustable):
        m, b, p = unpack(fit.x, adjustable)
        prediction, z = project(p, poses, m, K, vi, pi)
        errors = np.linalg.norm(prediction - uv, axis=1)
        closure = np.array(split['T_base_gripper']) @ m @ np.array(split['T_camera_target'])
        result = {}
        for label, mask in [('train', train), ('heldout', ~train)]:
            keep = np.array(split['split']) == label
            angle = Rotation.from_matrix(b[:3, :3].T @ closure[keep, :3, :3]).magnitude()
            result[label] = dict(corners=int(mask.sum()), rmse_px=float(np.sqrt(np.mean(errors[mask] ** 2))),
                p90_px=float(np.percentile(errors[mask], 90)), positive_depth=bool((z[mask] > 0).all()),
                closure_translation_rmse_mm=float(np.sqrt(np.mean(np.sum((closure[keep, :3, 3] - b[:3, 3]) ** 2, axis=1))) * 1000),
                closure_rotation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(angle ** 2)))))
        return result, m, b

    before, _, _ = metrics(baseline, False)
    after, fitted_mount, fitted_board = metrics(candidate, True)
    h, old = after['heldout'], before['heldout']
    passed = bool(candidate.success and not candidate.active_mask.any() and h['positive_depth']
        and h['closure_translation_rmse_mm'] <= 3 and h['closure_rotation_rmse_deg'] <= .5
        and h['closure_translation_rmse_mm'] <= .5 * old['closure_translation_rmse_mm']
        and h['closure_rotation_rmse_deg'] <= .5 * old['closure_rotation_rmse_deg'])
    private = output / 'private_candidate.json'
    save(private, dict(T_gripper_camera=fitted_mount.tolist(), T_base_board=fitted_board.tolist()))
    result = dict(experiment_id=output.name, result='success' if passed else 'failure',
        original_metric_gate_passed=passed, calibration_deployed=False, task_success=None,
        baseline=before, candidate=after, optimizer=dict(success=bool(candidate.success), nfev=candidate.nfev,
        active_bounds=candidate.active_mask.tolist()), camera_change=dict(
        translation_mm=float(np.linalg.norm(candidate.x[3:6])*1000), rotation_deg=float(np.rad2deg(np.linalg.norm(candidate.x[:3])))),
        train_phases=sorted(train_names), heldout_phases=sorted(heldout_names),
        private_candidate_sha256=digest(private), input_sha256={str(p.relative_to(source)):digest(p) for p in paths},
        program_sha256=digest(__file__), seconds=time.monotonic()-started, llm_calls=0, tokens=0,
        hardware_commands=0, visual_modality='real RGB only, measured joints and user-confirmed20mm board',
        limitations=['Same-session heldout views already examined in cycle8; not new independent validation',
                    'Nominal FK and factory RGB intrinsics fixed; no dynamics or contact identification',
                    'No change to existing acceptance thresholds or active transforms'])
    save(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))

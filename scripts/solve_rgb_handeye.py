"""Offline pose adapter for OpenCV Park-Martin; run with /usr/bin/python3.

T_a_b maps coordinates from b into a. Inputs are paired robot/FK poses and
RGB-derived metric target poses, with a target fixed in the robot base frame.
This consumes calibration observations, never changes the Episode format or
deploys a calibration. Pose provenance still needs independent verification.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from learn import digest, save


def motion_diagnostics(base_gripper):
    """Robot-side necessary conditions only; no visual accuracy or scale claim."""
    relative = np.array([np.linalg.inv(base_gripper[j])@base_gripper[i]
                         for i, j in combinations(range(len(base_gripper)), 2)])
    system = (relative[:, :3, :3]-np.eye(3)).reshape(-1, 3)
    singular = np.linalg.svd(system, compute_uv=False)
    rank = int(np.linalg.matrix_rank(system))
    return dict(pose_count=len(base_gripper), pair_count=len(relative), translation_system_rank=rank,
                singular_values=singular.tolist(),
                condition_number=float(singular[0]/singular[-1]) if rank==3 else None,
                rhs_error_amplification_bound=float(1/singular[-1]) if rank==3 else None,
                max_pair_rotation_deg=float(np.rad2deg(Rotation.from_matrix(relative[:,:3,:3]).magnitude()).max()),
                max_origin_separation_m=float(np.linalg.norm(relative[:,:3,3],axis=1).max()),
                interpretation='Necessary robot-motion check only. Pair residuals share observations; no independent-noise confidence interval. Does not establish monocular scale or visual pose quality.')


def solve(base_gripper, camera_target):
    motion = motion_diagnostics(base_gripper)
    if motion['translation_system_rank'] < 3:
        raise ValueError('Hand-eye translation is unobservable: static or parallel-axis motions')
    rotation, translation = cv2.calibrateHandEye(
        list(base_gripper[:, :3, :3]), list(base_gripper[:, :3, 3]),
        list(camera_target[:, :3, :3]), list(camera_target[:, :3, 3]),
        method=cv2.CALIB_HAND_EYE_PARK)
    result = np.eye(4); result[:3, :3] = rotation; result[:3, 3] = translation.ravel()
    if not np.isfinite(result).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise ValueError('OpenCV returned an invalid rigid transform; calibration remains unqualified')
    return result, np.array(motion['singular_values'])


def closure_errors(base_gripper, camera_target, mount, train):
    targets = base_gripper@mount@camera_target
    # The reference uses training poses only; evaluation cannot change the fit.
    reference_r = Rotation.from_matrix(targets[train, :3, :3]).mean()
    reference_t = np.mean(targets[train, :3, 3], axis=0)
    angles = np.rad2deg((reference_r.inv()*Rotation.from_matrix(targets[:, :3, :3])).magnitude())
    distance = np.linalg.norm(targets[:, :3, 3]-reference_t, axis=1)
    return {name:dict(frames=int(mask.sum()), translation_rmse_m=float(np.sqrt(np.mean(distance[mask]**2))),
                      rotation_rmse_deg=float(np.sqrt(np.mean(angles[mask]**2))))
            for name, mask in [('train',train),('heldout',~train)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    report = dict(result='infrastructure_error', task_success=None, calibration_promoted=False,
                  validation_status='UNQUALIFIED_POSE_CANDIDATE', opencv_version=cv2.__version__,
                  program_sha256=digest(__file__))
    try:
        report['source_sha256'] = digest(args.input)
        data = json.loads(args.input.read_text())
        if data['translation_unit'] != 'm':
            raise ValueError('Explicit meter-unit poses required; units are not guessed or converted')
        g = np.array(data['T_base_gripper'],float); c = np.array(data['T_camera_target'],float)
        train = np.array(data['split']) == 'train'
        if g.shape != c.shape or g.shape != (len(train),4,4) or train.sum()<3 or (~train).sum()<1 or not set(data['split']) <= {'train','heldout'}:
            raise ValueError('Paired 4x4 poses need at least three train poses and one heldout pose')
        old = np.array(data['baseline_T_gripper_camera'],float)
        for poses in (g,c,old[None]):
            r = poses[:,:3,:3]
            if not np.isfinite(poses).all() or not np.allclose(poses[:,3,:],[0,0,0,1]) or not np.allclose(r.transpose(0,2,1)@r,np.eye(3),atol=1e-6) or np.any(np.linalg.det(r)<0):
                raise ValueError('Input poses must be finite rigid transforms')
        report['robot_motion_diagnostics'] = motion_diagnostics(g[train])
        candidate, singular = solve(g[train],c[train])
        change = np.linalg.inv(old)@candidate
        translation = float(np.linalg.norm(change[:3,3])); angle = float(np.rad2deg(Rotation.from_matrix(change[:3,:3]).magnitude()))
        within = translation <= data['bounds']['max_translation_change_m'] and angle <= data['bounds']['max_rotation_change_deg']
        report.update(result='success' if within else 'failure',
            validation_status='POSE_CANDIDATE_ONLY' if within else 'REJECTED_OUTSIDE_REQUESTED_BOUNDS',
            T_gripper_camera=candidate.tolist(), parameter_change=dict(translation_m=translation,rotation_deg=angle),
            requested_bounds=data['bounds'], within_requested_bounds=bool(within),
            translation_system_singular_values=singular.tolist(),
            baseline=closure_errors(g,c,old,train), candidate=closure_errors(g,c,candidate,train),
            input_provenance=data['provenance'],
            limitations=['Closure errors are pose consistency, not independent physical ground truth',
                        'Full numerical rank does not establish adequate conditioning or observability under measurement noise',
                        'Metric target dimensions, RGB pose ambiguity, FK and timestamps must be verified upstream',
                        'No parameter promotion, hardware connection, or XML modification'])
    except (ValueError,cv2.error) as exc:
        report.update(result='failure',error=str(exc))
    except Exception as exc:
        report.update(result='infrastructure_error',error=f'{type(exc).__name__}: {exc}')
    finally:
        report['execution_time_s'] = time.monotonic()-started
        save(args.output/'result.json',report)
        (args.output/'REPORT.md').write_text('# Offline hand-eye pose adapter\n\n'+
            f"Result: {report['result']}; {report['validation_status']}. Calibration promoted: false.\n\n"+
            report.get('error','See result.json for baseline/candidate closure and source hash.')+'\n')
    return 0 if report['result']=='success' else 2


if __name__ == '__main__': raise SystemExit(main())

"""Inspect existing planar RGB motion candidates; never select a metric pose."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from learn import record, finish, digest


def candidates(homography, k, before, after, robot_angle_deg):
    count, rotations, translations, normals = cv2.decomposeHomographyMat(homography, k)
    p = cv2.undistortPoints(np.float32(before).reshape(-1,1,2), k, None)
    q = cv2.undistortPoints(np.float32(after).reshape(-1,1,2), k, None)
    visible = cv2.filterHomographyDecompByVisibleRefpoints(rotations,normals,p,q)
    ids = set() if visible is None else set(visible.ravel().tolist())
    rows=[]
    for i,(r,t) in enumerate(zip(rotations,translations)):
        angle=float(np.rad2deg(Rotation.from_matrix(r).magnitude()))
        rows.append(dict(index=i,visible_refpoint_filter_pass=i in ids,rotation_deg=angle,
                         robot_angle_difference_deg=abs(angle-robot_angle_deg),
                         translation_over_plane_distance_norm=float(np.linalg.norm(t))))
    return dict(candidate_count=count,visible_candidate_count=len(ids),candidates=rows,
                selected_candidate=None,metric_translation_available=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks',type=Path,required=True);p.add_argument('--robot-poses',type=Path,required=True)
    p.add_argument('--camera',type=Path,required=True);args=p.parse_args()
    out,report=record('diagnosis','Planar RGB motion ambiguity input audit','archived_rgb_geometry_audit')
    started=time.monotonic();rows=[]
    try:
        meta=json.loads(args.camera.read_text());poses=json.loads(args.robot_poses.read_text())
        if any(meta['distortion_coeffs']):
            raise ValueError('This audit uses zero-distortion RGB metadata only; no implicit inverse-Brown conversion')
        k=np.array(meta['K'],float);g=np.array(poses['T_base_gripper'])
        for i,frame in enumerate(poses['frame_indices'][1:],1):
            path=args.tracks/f'real_{frame}.npz';data=np.load(path);keep=data['survivors']
            angle=float(np.rad2deg(Rotation.from_matrix(g[0,:3,:3].T@g[i,:3,:3]).magnitude()))
            result=candidates(data['homography'],k,data['reference_px'][keep],data['target_px'][keep],angle)
            rows.append(dict(frame=frame,robot_rotation_deg=angle,source_sha256=digest(path),**result))
        report.update(result='success',task_success=None,validation_status='AMBIGUITY_AUDIT_ONLY',
                      expected_effects=['Preserve visible motion alternatives and check rotation-angle invariance'],
                      actions=['Existing OpenCV homography decomposition and visibility filter; no candidate selection'],
                      root_cause='UNKNOWN; local RGB motions retain ambiguity and do not supply metric target poses',
                      repair={'calibration_promoted':False,'physical_parameters_changed':False},
                      evidence=[{'kind':name,'sha256':digest(path)} for name,path in [('camera_metadata',args.camera),('robot_poses',args.robot_poses)]],
                      lessons_learned=['Visibility alone does not certify a motion candidate','Translation from decomposition is normalized by plane distance'],
                      next_improvement='Check motion consistency across views and uncertainty before any metric calibration',
                      new_physical_fit=False,new_research_round=False,stage2_rounds_used=6)
    except Exception as exc:report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        report['observed_effects']=rows;report['execution_time']=time.monotonic()-started
        report['tokens']={'planner':0,'research_orchestrator':None};report['human_interventions']=None
        report['analysis_version']=digest(__file__);finish(out,report)


if __name__=='__main__':main()

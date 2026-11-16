"""Metric RGB target observations for the existing hand-eye solver; never moves hardware."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from learn import digest,save

PATTERN=(11,8)  # User's12x9 squares, not12x9 inner corners.
SQUARE_M=.020  # User confirmed20mm on GP290-20-12*9.


def board_points():
    xy=np.indices(PATTERN).transpose(2,1,0).reshape(-1,2).astype(float)
    xy-= (np.array(PATTERN)-1)/2
    return np.column_stack([xy*SQUARE_M,np.zeros(len(xy))])


def pose_candidates(corners,K,distortion,points=None):
    points=board_points() if points is None else np.asarray(points,float)
    count,rotations,translations,_=cv2.solvePnPGeneric(points,np.asarray(corners,float),
        np.asarray(K,float),np.asarray(distortion,float),flags=cv2.SOLVEPNP_IPPE)
    candidates=[]
    for r,t in zip(rotations,translations):
        rotation=cv2.Rodrigues(r)[0]
        if np.min((points@rotation.T+t.reshape(1,3))[:,2])<=0: continue
        projected=cv2.projectPoints(points,r,t,np.asarray(K,float),np.asarray(distortion,float))[0]
        rms=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-corners.reshape(-1,2))**2,axis=1))))
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=t.ravel()
        candidates.append(dict(T_camera_target=pose.tolist(),reprojection_rms_px=rms))
    return sorted(candidates,key=lambda x:x['reprojection_rms_px'])


def track_corners(directory,start,stop,corners):
    """Track known grid identities through consecutive RGB; discard lost points.

    Uses OpenCV's lk_track.py forward/backward mechanism. No pose fit, no
    replacement points, and no extrapolation behind the gripper.
    """
    previous=cv2.imread(str(directory/f'{start:04d}.jpg'),cv2.IMREAD_GRAYSCALE)
    points=np.float32(corners).reshape(-1,1,2);ids=np.arange(len(points))
    options=dict(winSize=(21,21),maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for i in range(start+1,stop+1):
        current=cv2.imread(str(directory/f'{i:04d}.jpg'),cv2.IMREAD_GRAYSCALE)
        if previous is None or current is None:raise ValueError('Missing consecutive RGB frame')
        forward,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,**options)
        backward,back_status,_=cv2.calcOpticalFlowPyrLK(current,previous,forward,None,**options)
        valid=(status.ravel()!=0)&(back_status.ravel()!=0)&(np.linalg.norm(points-backward,axis=2).ravel()<.5)
        ids=ids[valid];points=forward[valid];previous=current
        if len(ids)<20:raise ValueError('Too few continuously observed grid corners')
    cv2.cornerSubPix(current,points,(3,3),(-1,-1),options['criteria'])
    _,inliers=cv2.findHomography(board_points()[ids,:2],points,cv2.RANSAC,2.)
    keep=inliers.ravel()!=0
    return ids[keep],points[keep].reshape(-1,2)


def observe(image,camera):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    found,corners=cv2.findChessboardCornersSB(gray,PATTERN,
        cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
    if not found:return dict(status='TARGET_NOT_FULLY_VISIBLE',corner_count=0,candidates=[])
    candidates=pose_candidates(corners,np.array(camera['K']),camera['distortion_coeffs'])
    return dict(status='TARGET_POSE_CANDIDATES' if candidates else 'NO_POSITIVE_DEPTH_SOLUTION',
        corner_count=len(corners),corners_px=corners.reshape(-1,2).tolist(),candidates=candidates,
        board=dict(inner_corners=list(PATTERN),square_m=SQUARE_M,origin='center of inner-corner lattice'),
        scope='Single-frame planar pose; preserve alternatives and verify a fixed target across heldout robot poses')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--camera',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    im=cv2.imread(str(a.image))
    if im is None: raise ValueError('Cannot read RGB image')
    result=observe(im,json.loads(a.camera.read_text()))
    result['input_sha256']={p.name:digest(p) for p in [a.image,a.camera]}
    save(a.output/'private_observation.json',result)
    if result['corner_count']:
        cv2.drawChessboardCorners(im,PATTERN,np.float32(result['corners_px']).reshape(-1,1,2),True)
        cv2.imwrite(str(a.output/'corners.png'),im)
    public={k:result[k] for k in ['status','corner_count','input_sha256']}
    public.update(reprojection_rms_px=[x['reprojection_rms_px'] for x in result['candidates']],
        private_observation_sha256=digest(a.output/'private_observation.json'),real_motion_commands=0)
    save(a.output/'result.json',public);print(json.dumps(public))

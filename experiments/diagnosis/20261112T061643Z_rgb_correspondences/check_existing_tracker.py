"""Reuse the existing tracker on fixed-fixture RGB; no parameter calibration."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import cv2
import numpy as np
from learn import DATA, save, digest
from validate_rgb_motion import track_step
from rgb_correspondences import observed_support


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    folder=Path(__file__).resolve().parent
    roi=json.loads((folder/'design.json').read_text())
    software=json.loads((folder/'software_design.json').read_text())
    capture=DATA/roi['capture']/'right_wrist'
    reference=cv2.imread(str(capture/f"{roi['reference_frame']:04d}.jpg"),0)
    mask=np.zeros_like(reference);cv2.fillPoly(mask,[np.array(roi['mask_polygon'],np.int32)],255)
    for x0,y0,x1,y1 in roi['exclude_rectangles']:mask[y0:y1,x0:x1]=0
    n=2*roi['mask_erode_px']+1;mask=cv2.erode(mask,np.ones((n,n),np.uint8))
    yy,xx=np.where(mask>0);grid=np.float32(np.column_stack([xx,yy]))[:,None,:]
    points=cv2.goodFeaturesToTrack(reference,30,.01,4,mask=mask).reshape(-1,2)
    cases=list(software['heldout_software_cases'])+[dict(id=f'real_{f}',frame=f) for f in roi['target_frames']]
    rows=[];started=time.monotonic()
    for case in cases:
        row=dict(id=case['id'],result='infrastructure_error',initial_features=len(points));t=time.monotonic()
        try:
            if 'H' in case:
                truth=np.array(case['H'],float)
                target=cv2.warpPerspective(reference,truth,(640,480))
                target=np.uint8(np.clip(target.astype(float)*case['gain']+case['offset'],0,255))
                row['type']='known_image_warp'
            else:
                path=capture/f"{case['frame']:04d}.jpg";target=cv2.imread(str(path),0)
                row.update(type='archived_real_pair',source_sha256=digest(path))
            q,status,fb=track_step(reference,target,points)
            keep=status & (fb<=1.)
            row['survivors']=int(keep.sum())
            if keep.sum()<4:raise ValueError('Fewer than four surviving tracks')
            cv2.setRNGSeed(0);h,inlier=cv2.findHomography(points[keep],q[keep],cv2.RANSAC,2.)
            if h is None:raise ValueError('No image homography')
            row.update(inliers=int(inlier.sum()),result='unverified',forward_backward_median_px=float(np.median(fb[keep])))
            measured=points[keep][inlier.ravel().astype(bool)]
            support=observed_support(mask,measured)
            row.update(interpolation_support_fraction=float((support>0).sum()/(mask>0).sum()),
                       inlier_bbox_px=[measured.min(0).tolist(),measured.max(0).tolist()])
            if 'H' in case:
                error=np.linalg.norm(cv2.perspectiveTransform(grid,h)-cv2.perspectiveTransform(grid,truth),axis=2).ravel()
                row.update(median_grid_error_px=float(np.median(error)),p90_grid_error_px=float(np.percentile(error,90)))
                row['result']='success' if np.median(error)<1. and np.percentile(error,90)<2. else 'failure'
            np.savez_compressed(args.output/f"{case['id']}.npz",reference_px=points,target_px=q,
                                survivors=keep,homography=h,forward_backward_px=fb)
        except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
        row['execution_time_s']=time.monotonic()-t;rows.append(row)
    report=dict(cases=rows,opencv=cv2.__version__,execution_time_s=time.monotonic()-started,
                tracker_source_sha256=digest(Path(__file__).resolve().parents[3]/'scripts/validate_rgb_motion.py'),
                run_script_sha256=digest(__file__),
                calibration_promoted=False,physical_parameter_fit=False,
                limitations=['No measured ground truth for real pairs; forward/backward consistency is not accuracy',
                             'Same source image and transforms already used to assess other methods; retrospective comparison',
                             'Existing 25x25 tracking support can cross the reference ROI and object boundaries'])
    save(args.output/'summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()

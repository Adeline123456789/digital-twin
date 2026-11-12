"""Known-image-warp checks, not physical calibration or robot episodes.

Use tote_insert Python with the isolated .local/loftr_runtime on PYTHONPATH.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import cv2
import numpy as np
import torch
import kornia
from learn import DATA, ROOT, digest, save
from loftr_correspondences import load_matcher, match_region
from rgb_correspondences import refine_plane


def error_stats(grid, estimated, truth):
    error=np.linalg.norm(cv2.perspectiveTransform(grid,estimated)-
                         cv2.perspectiveTransform(grid,truth),axis=2).ravel()
    return dict(median_grid_error_px=float(np.median(error)),
                p90_grid_error_px=float(np.percentile(error,90)),max_grid_error_px=float(error.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    folder=Path(__file__).resolve().parent
    design=json.loads((folder/'software_design.json').read_text())
    roi=json.loads((folder/'design.json').read_text())
    image_path=DATA/roi['capture']/'right_wrist'/f"{roi['reference_frame']:04d}.jpg"
    reference=cv2.imread(str(image_path),0)
    mask=np.zeros_like(reference);cv2.fillPoly(mask,[np.array(roi['mask_polygon'],np.int32)],255)
    for x0,y0,x1,y1 in roi['exclude_rectangles']:mask[y0:y1,x0:x1]=0
    n=2*roi['mask_erode_px']+1;mask=cv2.erode(mask,np.ones((n,n),np.uint8))
    yy,xx=np.where(mask>0);grid=np.float32(np.column_stack([xx,yy]))[:,None,:]
    checkpoint=ROOT/'.local/loftr_weights/loftr_indoor.ckpt'
    rows=[];started=time.monotonic()
    report=dict(result='infrastructure_error',cases=rows,physical_calibration=False,
                validation_status='SOFTWARE_KNOWN_WARP_ONLY',opencv=cv2.__version__,torch=torch.__version__,
                kornia=kornia.__version__,reference_sha256=digest(image_path),checkpoint_sha256=digest(checkpoint),
                criteria=design['criteria'],design_sha256=digest(folder/'software_design.json'))
    try:
        torch.set_num_threads(4);cv2.setNumThreads(4);model=load_matcher(checkpoint)
        for case in design['heldout_software_cases']:
            row=dict(id=case['id'],result='infrastructure_error');t=time.monotonic()
            try:
                truth=np.array(case['H'],float)
                target=cv2.warpPerspective(reference,truth,(reference.shape[1],reference.shape[0]))
                target=np.uint8(np.clip(target.astype(float)*case['gain']+case['offset'],0,255))
                p,q,confidence=match_region(model,reference,target,mask)
                row['matches']=len(p)
                if len(p)<4:raise ValueError('Fewer than four RGB matches')
                cv2.setRNGSeed(0);initial,inlier=cv2.findHomography(p,q,cv2.RANSAC,2.)
                if initial is None:raise ValueError('No image homography')
                row['inliers']=int(inlier.sum());row['before']=error_stats(grid,initial,truth)
                fitted,score=refine_plane(reference,target,mask,initial)
                row['after']=error_stats(grid,fitted,truth);row['ecc_score']=score
                criteria=design['criteria']
                row['result']='success' if (row['after']['median_grid_error_px']<criteria['median_grid_error_px_lt']
                    and row['after']['p90_grid_error_px']<criteria['p90_grid_error_px_lt']) else 'failure'
                np.savez_compressed(args.output/f"{case['id']}.npz",reference_px=p,target_px=q,
                                    confidence=confidence,initial_H=initial,refined_H=fitted)
            except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
            row['execution_time_s']=time.monotonic()-t;rows.append(row)
            save(args.output/'summary.json',report)
        report['result']='success' if all(r['result']=='success' for r in rows) else 'failure'
    except Exception as exc:report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        report['execution_time_s']=time.monotonic()-started
        save(args.output/'summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()

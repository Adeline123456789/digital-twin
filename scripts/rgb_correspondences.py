"""Extract planar RGB correspondences with installed OpenCV, not metric poses."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from learn import DATA, digest, save


def observed_support(mask, points):
    """Mark interpolation support; membership does not certify point accuracy."""
    support=np.zeros_like(mask)
    if len(points)>=3:
        hull=cv2.convexHull(np.asarray(points,np.float32))
        cv2.fillConvexPoly(support,np.rint(hull).astype(np.int32),255)
    return cv2.bitwise_and(support,mask)


def refine_plane(reference, target, reference_mask, initial):
    """ECC refines a reference->target image homography, never a metric pose."""
    # OpenCV's mask lives in INPUT (target) coordinates, not template coordinates.
    target_mask=cv2.warpPerspective(reference_mask,initial,
                                    (target.shape[1],target.shape[0]),flags=cv2.INTER_NEAREST)
    score,h=cv2.findTransformECC(reference,target,np.asarray(initial,np.float32).copy(),
                                cv2.MOTION_HOMOGRAPHY,
                                (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6),
                                target_mask,5)
    return h,float(score)


def match_plane(reference, target, mask):
    detector=cv2.SIFT_create()
    a,da=detector.detectAndCompute(reference,mask)
    b,db=detector.detectAndCompute(target,None)
    report=dict(reference_features=len(a),target_features=len(b),status='INSUFFICIENT_MATCHES')
    if da is None or db is None or min(len(a),len(b))<2:return report,{}
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def neighbors(x,y):
        return {p.queryIdx:p.trainIdx for p,q in matcher.knnMatch(x,y,k=2) if p.distance<.75*q.distance}
    forward=neighbors(da,db);backward=neighbors(db,da)
    ids=[(i,j) for i,j in forward.items() if backward.get(j)==i]
    p=np.array([a[i].pt for i,j in ids],np.float32).reshape(-1,2)
    q=np.array([b[j].pt for i,j in ids],np.float32).reshape(-1,2)
    # Multiple SIFT orientations at one location are not independent points.
    _,unique=np.unique(np.round(p,2),axis=0,return_index=True);p=p[unique];q=q[unique]
    report['mutual_unique_matches']=len(p)
    points=dict(reference_px=p.tolist(),target_px=q.tolist())
    if len(p)<4:return report,points
    cv2.setRNGSeed(0)
    h,inside=cv2.findHomography(p,q,cv2.RANSAC,2.)
    if h is None:return report,points
    inside=inside.ravel().astype(bool)
    predicted=cv2.perspectiveTransform(p[:,None,:],h)[:,0]
    residual=np.linalg.norm(predicted-q,axis=1)
    points.update(homography=h.tolist(),inlier=inside.tolist())
    area=cv2.contourArea(cv2.convexHull(p[inside]))
    report.update(status='IMAGE_CORRESPONDENCES_ONLY',inlier_count=int(inside.sum()),
                  inlier_fraction=float(inside.mean()),inlier_median_residual_px=float(np.median(residual[inside])),
                  inlier_max_residual_px=float(residual[inside].max()),reference_inlier_hull_area_px2=area)
    return report,points


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    design=json.loads(args.design.read_text());capture=DATA/design['capture']
    ref=capture/'right_wrist'/f"{design['reference_frame']:04d}.jpg"
    reference=cv2.imread(str(ref),cv2.IMREAD_GRAYSCALE)
    mask=np.zeros_like(reference);cv2.fillPoly(mask,[np.array(design['mask_polygon'],np.int32)],255)
    for x0,y0,x1,y1 in design['exclude_rectangles']:mask[y0:y1,x0:x1]=0
    erosion=design['mask_erode_px'];mask=cv2.erode(mask,np.ones((2*erosion+1,2*erosion+1),np.uint8))
    cv2.imwrite(str(args.output/'reference_mask.png'),mask)
    started=time.monotonic();summaries=[]
    for frame in design['target_frames']:
        target=capture/'right_wrist'/f'{frame:04d}.jpg'
        try:
            summary,points=match_plane(reference,cv2.imread(str(target),cv2.IMREAD_GRAYSCALE),mask)
            save(args.output/f'{frame:04d}_points.json',points)
            summary['points_sha256']=digest(args.output/f'{frame:04d}_points.json')
        except Exception as exc:
            summary=dict(status='INFRASTRUCTURE_ERROR',error=f'{type(exc).__name__}: {exc}')
        summaries.append(dict(frame=frame,target_sha256=digest(target),**summary))
    report=dict(opencv_version=cv2.__version__,design_sha256=digest(args.design),script_sha256=digest(__file__),
                reference_sha256=digest(ref),pairs=summaries,execution_time_s=time.monotonic()-started,
                root_cause_status='unknown',calibration_promoted=False,metric_scale_identified=False,
                limitations=['Image homography residuals are not independently measured correspondence accuracy',
                             'Mask is manually defined on reference RGB; descriptor support can cross its boundary',
                             'Planarity, semantic identities, metric dimensions and camera poses are not validated'])
    save(args.output/'summary.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

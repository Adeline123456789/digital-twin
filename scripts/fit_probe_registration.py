"""Offline FK-constrained RGB bundle adjustment; never deploys a calibration.

SHECAR Eq23 mechanism, implemented with installed OpenCV/SciPy (no copied code).
T_a_b maps b into a. Robot poses and intrinsics stay fixed. Heldout pixels do
not select landmarks or enter either baseline or candidate optimization.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from learn import DATA, SIM, digest, save


def public_summary(result_path):
    """Private transforms stay in the local fit artifact, not the public report."""
    report=json.loads(result_path.read_text())
    for key in ('T_link6_rgb','baseline_T_link6_rgb'):report.pop(key,None)
    report['private_calibration_evidence']={
        'dataset_relative_path':str(result_path.resolve().relative_to(DATA)) if result_path.resolve().is_relative_to(DATA) else None,
        'sha256':digest(result_path),
        'scope':'Device transforms retained locally; no parameter has been deployed'}
    return report


def project(points, poses, mount, K, vi, pi):
    camera = poses[vi] @ mount
    xyz = np.einsum('nji,nj->ni', camera[:, :3, :3], points[pi]-camera[:, :3, 3])
    pixels = xyz @ K.T
    return pixels[:, :2]/pixels[:, 2:], xyz[:, 2]


def triangulate(poses, mount, K, observations):
    projection = K @ np.linalg.inv(poses @ mount)[:, :3, :]
    points = []
    for obs in observations:
        rows = [row for view, uv in obs.items() for row in
                (uv[0]*projection[view, 2]-projection[view, 0],
                 uv[1]*projection[view, 2]-projection[view, 1])]
        _, _, v = np.linalg.svd(rows)
        points.append(v[-1, :3]/v[-1, 3])
    return np.array(points)


def mutual_matches(d0, d1):
    matcher = cv2.BFMatcher()
    def ratio(a, b):
        return {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(a,b,k=2) if m.distance<.70*n.distance}
    forward, backward = ratio(d0,d1), ratio(d1,d0)
    return [(a,b) for a,b in forward.items() if backward.get(b)==a]


def lk_matches(first, second, points):
    target,valid,error=cv2.calcOpticalFlowPyrLK(first,second,points,None,winSize=(31,31),maxLevel=4,
                                              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    back,reverse,_=cv2.calcOpticalFlowPyrLK(second,first,target,None,winSize=(31,31),maxLevel=4,
                                          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    good=(valid.ravel()>0)&(reverse.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<.5)&(error.ravel()<20)
    uv=target.reshape(-1,2)
    good&=(uv[:,0]>=150)&(uv[:,0]<450)&(uv[:,1]>=20)&(uv[:,1]<350)
    return target,good


def collect(data, method='sift'):
    cv2.setRNGSeed(0)
    views, points, labels, counts = [], [], [], {}
    sift = cv2.SIFT_create(nfeatures=1000)
    for region in ('holder','socket'):
        selected = [next(v for v in data['views'] if v['phase']==f'{region}_{tag}_hold')
                    for tag in ('center','j4','j5','j6','translation','heldout')]
        offset = len(views); views.extend(selected)
        features=[];images=[]
        for view in selected:
            gray=cv2.imread(view['image_path'],cv2.IMREAD_GRAYSCALE)
            if gray is None:raise ValueError('Missing RGB image: '+view['image_path'])
            mask=np.zeros_like(gray);mask[20:350,150:450]=255
            k,d=sift.detectAndCompute(gray,mask);features.append((k,d));images.append(gray)
        k0,d0=features[0]
        anchor=(np.float32([k.pt for k in k0]).reshape(-1,1,2) if method=='sift' else
                cv2.goodFeaturesToTrack(images[0],maxCorners=150,qualityLevel=.005,minDistance=8,mask=mask,blockSize=5))
        tracks={i:{offset:pt.copy()} for i,pt in enumerate(anchor.reshape(-1,2))}
        counts[region]={'anchor_features':len(anchor),'matched_views':{}}
        for j,(k,d) in enumerate(features[1:],1):
            if method=='sift':
                pairs=mutual_matches(d0,d) if d0 is not None and d is not None else []
                ids=np.array([i for i,_ in pairs],dtype=int)
                a=np.float32([k0[i].pt for i,_ in pairs]);b=np.float32([k[t].pt for _,t in pairs])
            else:
                target,good=lk_matches(images[0],images[j],anchor)
                ids=np.flatnonzero(good);a=anchor[good].reshape(-1,2);b=target[good].reshape(-1,2)
            if len(a)<8:
                counts[region]['matched_views'][selected[j]['phase']]=0;continue
            _,inlier=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,.8,.999)
            if inlier is None:raise ValueError('No geometric verification for '+selected[j]['phase'])
            for i,uv,ok in zip(ids,b,inlier.ravel()):
                if ok:tracks[i][offset+j]=uv.copy()
            counts[region]['matched_views'][selected[j]['phase']]=int(inlier.sum())
        for track in tracks.values():
            if sum(v<offset+5 for v in track)>=4:
                points.append(track);labels.append(region)
    return views,points,labels,counts


def unpack(x, mount, adjustable):
    if not adjustable:return mount,x.reshape(-1,3)
    result=mount.copy()
    result[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@mount[:3,:3]
    result[:3,3]+=x[3:6]
    return result,x[6:].reshape(-1,3)


def fit(points, poses, mount, K, vi, pi, uv, adjustable):
    n=len(points);head=6 if adjustable else 0
    x=np.r_[np.zeros(head),points.ravel()]
    sparsity=lil_matrix((2*len(vi),len(x)),dtype=int)
    sparsity[:,:head]=1
    for row,index in enumerate(pi):sparsity[2*row:2*row+2,head+3*index:head+3*index+3]=1
    low=np.full(len(x),-np.inf);high=-low
    if adjustable:
        high[:6]=[np.deg2rad(25)]*3+[.075]*3;low[:6]=-high[:6]
    def residual(x):
        m,p=unpack(x,mount,adjustable)
        return (project(p,poses,m,K,vi,pi)[0]-uv).ravel()
    result=least_squares(residual,x,jac_sparsity=sparsity.tocsr(),bounds=(low,high),
                         x_scale='jac',loss='soft_l1',f_scale=1,max_nfev=100,
                         ftol=1e-8,xtol=1e-8,gtol=1e-8)
    return result,unpack(result.x,mount,adjustable)


def metrics(points,poses,mount,K,vi,pi,uv,labels,train):
    predicted,depth=project(points,poses,mount,K,vi,pi)
    error=np.linalg.norm(predicted-uv,axis=1)
    masks={'train':train,'heldout':~train}
    for region in ('holder','socket'):
        masks[region+'_heldout']=(np.array(labels)[pi]==region)&~train
    return {name:dict(observations=int(mask.sum()),median_px=float(np.median(error[mask])),
                      p90_px=float(np.percentile(error[mask],90)),
                      rmse_px=float(np.sqrt(np.mean(error[mask]**2))),
                      positive_depth=bool(np.all(depth[mask]>0)))
            for name,mask in masks.items() if mask.any()}


def run(args):
    started=time.monotonic()
    data=json.loads(args.input.read_text());K=np.array(data['K'])
    initializer=SIM/'configs/calibration/plug_rgbd_20261108.json'
    mount=np.array(json.loads(initializer.read_text())['cameras']['right_wrist']['T_parent_rgb'])
    views,tracks,labels,counts=collect(data,args.features);poses=np.array([v['T_base_link6'] for v in views])
    train_view=np.array([not v['phase'].endswith('_heldout_hold') for v in views])
    training=[{i:uv for i,uv in tr.items() if train_view[i]} for tr in tracks]
    points=triangulate(poses,mount,K,training);keep=[]
    for i,tr in enumerate(training):
        vi=np.array(list(tr));pi=np.full(len(tr),i);uv=np.array(list(tr.values()))
        pred,z=project(points,poses,mount,K,vi,pi)
        keep.append(bool(np.all((z>.03)&(z<1)) and np.sqrt(np.mean(np.sum((pred-uv)**2,axis=1)))<15))
    tracks=[t for t,k in zip(tracks,keep) if k];labels=[t for t,k in zip(labels,keep) if k];points=points[keep]
    rows=[(v,i,*uv) for i,tr in enumerate(tracks) for v,uv in tr.items()]
    if len(points)<12:raise ValueError(f'Only {len(points)} usable training landmarks; no fit')
    rows=np.array(rows);vi=rows[:,0].astype(int);pi=rows[:,1].astype(int);uv=rows[:,2:];train=train_view[vi]
    save(args.output/'observations.json',dict(views=views,tracks=[{str(k):v.tolist() for k,v in t.items()} for t in tracks],labels=labels,counts=counts))
    print('LANDMARKS',len(points),'OBSERVATIONS',len(rows),'COUNTS',counts,flush=True)
    baseline,(bm,bp)=fit(points,poses,mount,K,vi[train],pi[train],uv[train],False)
    candidate,(cm,cp)=fit(bp,poses,mount,K,vi[train],pi[train],uv[train],True)
    b=metrics(bp,poses,bm,K,vi,pi,uv,labels,train);c=metrics(cp,poses,cm,K,vi,pi,uv,labels,train)
    delta=candidate.x[:6];translation=float(np.linalg.norm(delta[3:]));rotation=float(np.rad2deg(np.linalg.norm(delta[:3])))
    accepted=(candidate.success and translation<=.075 and rotation<=25 and not candidate.active_mask[:6].any()
              and all(key in c and c[key]['median_px']<=2 and c[key]['p90_px']<=4 and c[key]['positive_depth']
                      for key in ('holder_heldout','socket_heldout'))
              and c['heldout']['median_px']<=.7*b['heldout']['median_px'])
    result=dict(experiment_id=args.output.name,timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                result='success' if accepted else 'failure',status='RESERVED_VIEW_CONSISTENCY_ONLY' if accepted else 'REJECTED',
                calibration_promoted=False,hardware_commands=0,real_depth=False,task_success=None,
                initializer_sha256=digest(initializer),source_sha256=digest(args.input),program_sha256=digest(__file__),feature_method=args.features,
                baseline=b,candidate=c,parameter_change=dict(translation_m=translation,rotation_deg=rotation),
                T_link6_rgb=cm.tolist(),baseline_T_link6_rgb=bm.tolist(),counts=counts,landmarks=len(points),
                optimizer=dict(success=bool(candidate.success),message=candidate.message,nfev=int(candidate.nfev),
                               active_bounds=candidate.active_mask[:6].tolist()),
                seconds=time.monotonic()-started,tokens=0,llm_calls=0,
                limitations=['Metric scale relies on nominal FK and factory intrinsics, not external metrology',
                             'Same-session reserved views; no independent physical TCP or object validation',
                             'No depth, force, dynamics, hardware or simulator configuration modified'])
    np.savez(args.output/'reconstruction.npz',points=cp,baseline_points=bp,vi=vi,pi=pi,uv=uv,train=train,poses=poses,K=K,mount=cm)
    save(args.output/'result.json',result)
    public=public_summary(args.output/'result.json')
    save(args.output/'public_result.json',public)
    (args.output/'REPORT.md').write_text('# Offline RGB registration\n\n'+
        f"Result: {result['result']}; {result['status']}. Calibration promoted: false.\n\n"+
        'See result.json for measured baseline/candidate errors, provenance and limits. No hardware commands were sent.\n')
    print(json.dumps(public,indent=2))
    return 0 if accepted else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--features',choices=['sift','lk'],default='sift')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        return run(args)
    except Exception as exc:
        # Insufficient matches are an expected experimental outcome, not lost evidence.
        save(args.output/'result.json',dict(experiment_id=args.output.name,
             timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             result='failure' if isinstance(exc,ValueError) else 'infrastructure_error',
             error=f'{type(exc).__name__}: {exc}',calibration_promoted=False,task_success=None,
             hardware_commands=0,program_sha256=digest(__file__)))
        (args.output/'REPORT.md').write_text('# Offline RGB registration failed\n\n'+
            f'{type(exc).__name__}: {exc}\n\nNo calibration was deployed. See result.json.\n')
        raise


if __name__=='__main__':raise SystemExit(main())

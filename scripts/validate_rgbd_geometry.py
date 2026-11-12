"""Offline cross-view object consistency and visible finger-surface depth checks."""
import csv
import json
import sys
import time
import cv2
import mujoco
import numpy as np
import pyrealsense2 as rs
import yaml
from audit_rgbd_evidence import yellow_mask,depth_stats
from fit_replay_timing import PAIRS
from learn import DATA,SIM,ROOT,record,finish,digest


def intrinsics(meta):
    k=np.array(meta['K']);i=rs.intrinsics();i.width,i.height=meta['resolution']
    i.fx=k[0,0];i.fy=k[1,1];i.ppx=k[0,2];i.ppy=k[1,2]
    i.model=getattr(rs.distortion,meta['distortion_model'].split('.')[-1]);i.coeffs=meta['distortion_coeffs']
    return i


def mask_distance(pixels,mask):
    indices=np.rint(pixels).astype(int)
    inside=(indices[:,0]>=0)&(indices[:,0]<mask.shape[1])&(indices[:,1]>=0)&(indices[:,1]<mask.shape[0])
    distances=cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    d=distances[indices[inside,1],indices[inside,0]]
    return {'in_image_fraction':float(inside.mean()),'median_px':float(np.median(d)) if len(d) else None,
            'p95_px':float(np.percentile(d,95)) if len(d) else None}


def finger_surfaces(image):
    """Visible dark finger top surfaces only, not inner pad/contact-face locations."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);points=[]
    for x0,x1 in ((80,280),(440,640)):
        mask=np.zeros(gray.shape,np.uint8);mask[350:480,x0:x1]=(gray[350:480,x0:x1]<65)*255
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        if n==1:points.append(None);continue
        label=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
        interior=cv2.erode((labels==label).astype('uint8'),np.ones((7,7),np.uint8))
        y,x=np.where(interior)
        points.append([int(np.median(x[y==y.min()])),int(y.min())] if len(y) else None)
    return points


def main():
    design=json.loads((ROOT/'experiments/research_stage2/round5/design.json').read_text())
    if not design.get('depth_provenance_verified',False):
        raise SystemExit('Stopped: historical depth provenance is unverified; user reports no real RGB-D.')
    out,r=record('research_stage2','Cross-view object and finger geometry validation','real_historical')
    started=time.monotonic();raw=DATA/'calibration'/r['experiment_id'];raw.mkdir()
    design=json.loads((ROOT/'experiments/research_stage2/round5/design.json').read_text());g=design['criteria']
    effects=[];evidence=[];sim=None
    try:
        sys.path.insert(0,str(SIM/'src'))
        from piper_sim.config import load
        from piper_sim.backends.mujoco import MujocoBackend
        task_path=SIM/'configs/tasks/real_connector_calibrated.yaml';task=yaml.safe_load(task_path.read_text())
        camera_path=SIM/task['camera_calibration']
        sim=MujocoBackend(dict(load(),task_scene=task))
        r['initial_state']={'camera_profile':'original provisional profile, unchanged','camera_sha256':digest(camera_path),
                            'task_sha256':digest(task_path),'design_sha256':digest(ROOT/'experiments/research_stage2/round5/design.json')}
        for name,_ in PAIRS:
            capture=DATA/'complete_real/failed'/name/'raw/task'
            command=capture/'commanded_probe.csv';rows=list(csv.DictReader(command.open()))
            ct=np.array([float(x['monotonic']) for x in rows]);joints=np.array([[float(x[f'feedback_j{k}']) for k in range(1,7)] for x in rows])
            events=[]
            for phase,fraction in [('grasp',0),('grasp',1),('lift',.5),('lift',1)]:
                segment=[x for x in rows if x['phase']==phase]
                events.append((phase,fraction,float(segment[int((len(segment)-1)*fraction)]['monotonic'])))
            selected={}
            for camera in ('external','right_wrist'):
                p=capture/camera;meta=json.loads((p/'camera.json').read_text())
                if meta['depth_frame']!='aligned to RGB optical frame':raise ValueError('Unaligned depth')
                ts=np.array([x['rgb_device_ms']/1000+x['host_monotonic']-x['host_unix'] for x in meta['samples']])
                ids=[int(np.argmin(abs(ts-t))) for _,_,t in events]
                with np.load(p/'depth.npz') as a:depth=a['depth_raw'][ids].copy()
                selected[camera]=(meta,ts,ids,depth)
                evidence.extend({'dataset_relative_path':str(f.relative_to(DATA)),'sha256':digest(f)} for f in (p/'camera.json',p/'depth.npz'))
            evidence.append({'dataset_relative_path':str(command.relative_to(DATA)),'sha256':digest(command)})
            for k,(phase,fraction,_) in enumerate(events):
                em,et,ei,ed=selected['external'];wm,wt,wi,wd=selected['right_wrist']
                ep=capture/f'external/{ei[k]:04d}.jpg';wp=capture/f'right_wrist/{wi[k]:04d}.jpg'
                external=cv2.imread(str(ep));wrist=cv2.imread(str(wp))
                mask=yellow_mask(external,(450,220,580,310));wrist_mask=yellow_mask(wrist,(250,300,500,480))
                y,x=np.where(mask&(ed[k]>0))
                cloud=np.array([rs.rs2_deproject_pixel_to_point(intrinsics(em),[float(u),float(v)],float(ed[k,v,u]*em['depth_scale_m'])) for v,u in zip(y,x)])
                angles=[np.interp(wt[wi[k]],ct,joints[:,j]) for j in range(6)]
                # Measured-pose FK only: no commands, integration, or safety-rule edits.
                sim.data.qpos[sim.indices['right']['qpos'][:6]]=np.deg2rad(angles);mujoco.mj_forward(sim.model,sim.data)
                world=sim.data.camera('overhead');hand=sim.data.camera('right_wrist')
                er=world.xmat.reshape(3,3)@np.diag([1.,-1.,-1.]);wr=hand.xmat.reshape(3,3)@np.diag([1.,-1.,-1.])
                wrist_cloud=(cloud@er.T+world.xpos-hand.xpos)@wr
                front=wrist_cloud[:,2]>0
                pixels=np.array([rs.rs2_project_point_to_pixel(intrinsics(wm),p.tolist()) for p in wrist_cloud[front]])
                distance=mask_distance(pixels,wrist_mask) if len(pixels) else {'in_image_fraction':0.,'median_px':None,'p95_px':None}
                obj=depth_stats(ed[k],mask,em['depth_scale_m']);dt=abs(float(et[ei[k]]-wt[wi[k]]))
                ok=(obj['positive_fraction']>=g['external_target_positive_depth_fraction_min'] and len(cloud)>=g['external_target_positive_points_min']
                    and dt<=g['camera_pair_time_difference_s_max'] and front.all() and distance['in_image_fraction']>=g['projected_in_image_fraction_min']
                    and distance['median_px'] is not None and distance['median_px']<=g['median_distance_to_wrist_target_mask_px_max']
                    and distance['p95_px']<=g['p95_distance_px_max'])
                fingers=[];overlay=wrist.copy()
                for point in finger_surfaces(wrist):
                    if point is None:fingers.append({'point':None,'depth_usable':False});continue
                    u,v=point;patch=wd[k,v-2:v+3,u-2:u+3];values=patch[patch>0]*wm['depth_scale_m']
                    usable=len(values)>=20 and np.std(values)<=g['finger_patch_std_m_max']
                    fingers.append({'point':point,'positive_fraction':len(values)/25,'std_m':float(np.std(values)) if len(values) else None,
                                    'median_m':float(np.median(values)) if len(values) else None,'depth_usable':bool(usable)})
                    cv2.circle(overlay,(u,v),5,(0,0,255),1)
                for u,v in np.rint(pixels).astype(int):
                    if 0<=u<640 and 0<=v<480:cv2.circle(overlay,(u,v),1,(255,0,255),-1)
                cv2.imwrite(str(raw/f'{name}_{phase}_{fraction}.jpg'),overlay)
                effects.append({'source_episode':name,'phase':phase,'fraction':fraction,'external_frame':ei[k],'wrist_frame':wi[k],
                    'camera_pair_delta_s':dt,'external_object_depth':obj,'object_projection':distance,'object_consistency_pass':bool(ok),
                    'finger_surface_patches':fingers,'metric_inner_aperture_mm':None})
                evidence.extend({'dataset_relative_path':str(p.relative_to(DATA)),'sha256':digest(p)} for p in (ep,wp))
        r.update(result='success',task_success=None,validation_status='GEOMETRY_AUDIT_ONLY',observed_effects=effects,evidence=evidence,
                 expected_effects=['Validate measured geometry before any contact fit'],actions=[design],
                 repair={'physics_changed':False,'calibration_promoted':False},
                 root_cause='UNKNOWN: camera/FK, occlusion, surface correspondence and depth errors remain possible',
                 lessons_learned=['Projected surface agreement is a necessary consistency check, not a full object pose or calibrated aperture'],
                 next_improvement='Stop physical fitting if geometry evidence fails; retain raw observations and numerical audit')
        r['limitations']=['Only original provisional camera/FK profile assessed','Cross-view visibility differs; silhouette distance is not a unique extrinsic error',
            'Finger surface points are not inner pad landmarks; no aperture estimate or contact force invented',
            'Same task/installation, eight paired historical keyframes; no broad statistical claim']
    except Exception as exc:r['error']=str(exc)
    finally:
        if sim is not None:sim.close()
        r['analysis_version']=digest(__file__);r['tokens']={'planner':0,'research_orchestrator':None}
        r['execution_time']=time.monotonic()-started;r['simulation_seconds']=0;finish(out,r)


if __name__=='__main__':main()

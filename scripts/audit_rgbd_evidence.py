"""Check existing RGB-D measurement support before attempting contact identification."""
import csv
import json
import time
import cv2
import numpy as np
from learn import DATA,record,finish,digest
from fit_replay_timing import PAIRS


def yellow_mask(image,roi=None):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([15,85,45]),np.array([42,255,255]))
    if roi is not None:
        x0,y0,x1,y1=roi;region=np.zeros(mask.shape,np.uint8);region[y0:y1,x0:x1]=255
        mask=cv2.bitwise_and(mask,region)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=10]
    if not candidates:return np.zeros(mask.shape,bool)
    largest=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    return cv2.erode((labels==largest).astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)


def depth_stats(depth,mask,scale):
    values=depth[mask];positive=values[values>0]*scale
    return {'mask_pixels':int(mask.sum()),'positive_depth_pixels':len(positive),
            'positive_fraction':float(len(positive)/len(values)) if len(values) else None,
            'depth_p05_median_p95_m':np.percentile(positive,[5,50,95]).tolist() if len(positive) else None}


def main():
    out,r=record('research_stage2','RGB-D evidence availability at grasp/lift transitions','real_historical')
    start=time.monotonic();observations=[];evidence=[]
    try:
        for name,_ in PAIRS:
            capture=DATA/'complete_real/failed'/name/'raw/task'
            command=capture/'commanded_probe.csv';rows=list(csv.DictReader(command.open()))
            events=[]
            for phase,fraction in [('grasp',0),('grasp',1),('lift',.5),('lift',1)]:
                segment=[row for row in rows if row['phase']==phase]
                row=segment[int((len(segment)-1)*fraction)]
                events.append((phase,fraction,float(row['monotonic']),float(row['feedback_gripper_open'])*95))
            evidence.append({'dataset_relative_path':str(command.relative_to(DATA)),'sha256':digest(command)})
            # Bounds selected from training-episode views only; limited to these grasp/lift frames.
            for camera,roi in (('right_wrist',(250,300,500,480)),('external',(450,220,580,310))):
                directory=capture/camera;meta_path=directory/'camera.json';depth_path=directory/'depth.npz'
                meta=json.loads(meta_path.read_text())
                if meta['depth_frame']!='aligned to RGB optical frame':raise ValueError('RGB/depth coordinates differ')
                stamps=np.array([s['rgb_device_ms']/1000+s['host_monotonic']-s['host_unix'] for s in meta['samples']])
                with np.load(depth_path) as archive:
                    depths=archive['depth_raw']
                    for phase,fraction,t,width in events:
                        i=int(np.argmin(abs(stamps-t)));image_path=directory/f'{i:04d}.jpg'
                        image=cv2.imread(str(image_path));mask=yellow_mask(image,roi)
                        row={'source_episode':name,'camera':camera,'phase':phase,'phase_fraction':fraction,
                             'frame':i,'exposure_minus_command_s':float(stamps[i]-t),'encoder_width_mm':width,
                             'target_roi_xyxy':roi,
                             'yellow_component_depth':depth_stats(depths[i],mask,meta['depth_scale_m'])}
                        if camera=='right_wrist':
                            # Fixed inspection ROIs, not claimed semantic fingertip segmentation.
                            row['finger_roi_depth']={}
                            for label,region in [('left',(slice(400,475),slice(130,230))),('right',(slice(400,475),slice(490,590)))]:
                                finger_roi=np.zeros(mask.shape,bool);finger_roi[region]=True
                                row['finger_roi_depth'][label]=depth_stats(depths[i],finger_roi,meta['depth_scale_m'])
                        observations.append(row)
                        evidence.append({'dataset_relative_path':str(image_path.relative_to(DATA)),'sha256':digest(image_path)})
                evidence.extend({'dataset_relative_path':str(p.relative_to(DATA)),'sha256':digest(p)} for p in (meta_path,depth_path))
        r.update(result='success',task_success=None,validation_status='OBSERVABILITY_AUDIT_NOT_CALIBRATION',
            expected_effects=['Check whether independently measured aperture/object geometry supports physical fitting'],
            observed_effects=observations,evidence=evidence,
            root_cause='UNKNOWN; RGB/depth availability alone does not determine contact physics',
            lessons_learned=['Nonzero depth is not a verified object surface; missing/occluded data cannot be replaced with simulator ground truth'],
            next_improvement='Verify actual surface correspondences and camera geometry before accepting any physical fit')
        r['limitations']=['ROIs are specific to these fixed-camera grasp/lift views, not general object recognition',
            'Left wrist excluded: inspected transition view does not contain the target; global yellow segmentation found unrelated objects',
            'Fixed finger rectangles can contain background and are not aperture measurements',
            'Only selected transition keyframes tested, not every frame or every calibration source',
            'No metric object pose, real contact force, friction, mass or COM inferred']
    except Exception as exc:r['error']=str(exc)
    finally:
        r['analysis_version']=digest(__file__);r['tokens']={'planner':0,'research_orchestrator':None}
        r['execution_time']=time.monotonic()-start;r['simulation_seconds']=0;finish(out,r)


if __name__=='__main__':main()

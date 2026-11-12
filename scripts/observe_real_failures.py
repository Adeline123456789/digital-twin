"""Read-only wrist-video change detection, aligned using recorded monotonic clocks."""
import csv
import json
import time
import cv2
import h5py
import numpy as np
from learn import DATA,ROOT,record,finish,digest,save

def camera_monotonic_times(archive, camera):
    """Map each RGB exposure to the exporter's host clock, preserving frame order."""
    return (archive['timestamp_monotonic'][:] +
            (archive[f'camera/{camera}/timestamp'][:] - archive['timestamp'][:]))

def yellow_top(frame,saturation):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([15,saturation,45]),np.array([42,255,255]))
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
    candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=200 and stats[i,cv2.CC_STAT_WIDTH]>=20]
    if not candidates:return None
    i=max(candidates,key=lambda k:stats[k,cv2.CC_STAT_AREA])
    return int(stats[i,cv2.CC_STAT_TOP])

def main():
    for name in ('20261109_real_smooth_attempt_000001','20261109_real_residual_attempt_000001','20261109_real_grip_hold_attempt_000001'):
        out,r=record('diagnosis','Wrist visual-change localization '+name,'real_historical')
        start=time.monotonic();ep=DATA/'complete_real/failed'/name
        try:
            video=ep/'videos/cam_right_wrist_rgb.mp4';h5=ep/'proprio_stats/proprio_stats.h5';csvfile=ep/'raw/task/commanded_probe.csv'
            commands=list(csv.DictReader(csvfile.open()));ct=np.array([float(x['monotonic']) for x in commands])
            with h5py.File(h5) as f:
                ts=camera_monotonic_times(f,'cam_right_wrist')
                state=f['state/right_arm/joint_positions'][:];action=f['action/right_arm/joint_positions'][:]
                valid=f['valid/right_arm_joint_action'][:].astype(bool)&f['valid/state'][:].astype(bool)
            indices=np.clip(np.searchsorted(ct,ts,side='right')-1,0,len(ct)-1)
            phases=[commands[i]['phase'] for i in indices]
            cap=cv2.VideoCapture(str(video));series={s:[] for s in (65,85,105)};i=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                for s in series:series[s].append(yellow_top(frame,s))
                i+=1
            cap.release()
            if i!=len(ts):raise ValueError('Video/HDF5 frame count mismatch')
            candidates=[]
            for saturation,ys in series.items():
                lift=[i for i,p in enumerate(phases) if p=='lift' and ys[i] is not None][-10:]
                if len(lift)<5:raise ValueError('Insufficient visible lift baseline')
                baseline=float(np.median([ys[i] for i in lift]));candidate=None
                for i in range(lift[-1]+1,len(ts)-2):
                    if phases[i] not in ('transport','align','insert'):continue
                    if all(ys[j] is not None and abs(ys[j]-baseline)>8 for j in range(i,i+3)):
                        candidate={'frame_zero_based':i,'phase':phases[i],'time_from_start_s':float(ts[i]-ts[0]),'top_y':ys[i],'baseline_y':baseline,'delta_y':ys[i]-baseline};break
                candidates.append({'saturation_min':saturation,'candidate':candidate})
            source=json.loads((ep/'raw/task/execution_report.json').read_text())
            r.update(result='success',task_success=False,validation_status='OBSERVATION_ANALYZED_NOT_CAUSE_VERIFIED',
                observed_effects=candidates,root_cause='UNKNOWN: image appearance change can reflect slip, contact, rotation or occlusion',
                earliest_observed_anomaly={'candidates':candidates,'definition':'First three consecutive frames beyond 8 px from lift-end median, only within transport/align/insert'},
                execution_stop_frame=source.get('insertion_stopped_frame'),
                joint_tracking_rmse_recorded_units=np.sqrt(np.mean((state[valid]-action[valid])**2,axis=0)).tolist(),
                lessons_learned=['Detect visual change before treating the controller stop frame as anomaly onset.',
                                 'Yellow segmentation is not calibrated 3D pose or force sensing.'],
                next_improvement='Validate candidate frames manually and against independent object pose; separate occlusion/contact/slip')
            r['evidence']=[{'dataset_relative_path':str(p.relative_to(DATA)),'sha256':digest(p)} for p in (video,h5,csvfile)]
            r['human_interventions']=None;r['robot_execution_seconds']=None
            r['limitations']=['Threshold-based yellow tracking may include unrelated yellow objects',
                'Three saturation thresholds are a sensitivity test, not independent ground truth',
                'No new real motion or physical calibration; raw video never uploaded']
            save(out/'frame_features.json',{'top_y_by_saturation':series,'phase_by_frame':phases})
        except Exception as exc:r['error']=str(exc)
        finally:r['execution_time']=time.monotonic()-start;finish(out,r)
if __name__=='__main__':main()

"""Reuse the existing plane fitter and validate observable camera geometry across episodes."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import time
import numpy as np
import pyrealsense2 as rs
import yaml
from learn import DATA,SIM,record,finish,digest,save

EPISODES=('20261109_real_residual_attempt_000001',
          '20261109_real_grip_hold_attempt_000001',
          '20261109_real_smooth_attempt_000001')
FRAMES=[0,30,60]


def measured_cloud(meta,depth):
    k=np.array(meta['K']);intr=rs.intrinsics();intr.width=640;intr.height=480
    intr.fx=k[0,0];intr.fy=k[1,1];intr.ppx=k[0,2];intr.ppy=k[1,2]
    intr.model=getattr(rs.distortion,meta['distortion_model'].split('.')[-1]);intr.coeffs=meta['distortion_coeffs']
    v,u=np.mgrid[:480,:640]
    mask=((v>160)&(v<230)&(u>55)&(u<590))|((v>305)&(v<355)&(u>190)&(u<430))
    mask&=depth>0
    y,x=np.where(mask)
    rays=np.array([rs.rs2_deproject_pixel_to_point(intr,[float(px),float(py)],1.) for py,px in zip(y,x)])
    return rays*(depth[y,x]*meta['depth_scale_m'])[:,None]


def plane_error(cloud,transform):
    return cloud@transform[2,:3]+transform[2,3]-.025


def metrics(error):
    return {'samples':len(error),'rmse_mm':float(np.sqrt(np.mean(error**2))*1000),
            'median_abs_mm':float(np.median(abs(error))*1000),
            'p95_abs_mm':float(np.percentile(abs(error),95)*1000)}


def main():
    out,r=record('research_stage2','Existing table/camera fit with separate real-episode validation','real_sim_calibration')
    started=time.monotonic();raw=DATA/'calibration'/r['experiment_id'];raw.mkdir()
    try:
        fitter=SIM/'scripts/fit_plug_table_depth.py'
        basepath=SIM/'configs/calibration/plug_rgbd_20261108.json'
        calibration=json.loads(basepath.read_text());old=np.array(calibration['cameras']['overhead']['T_parent_rgb'])
        source=DATA/'complete_real/failed'/EPISODES[0]/'raw/task/external'
        metadata=json.loads((source/'camera.json').read_text())
        with np.load(source/'depth.npz') as a:training_depth=a['depth_raw'][FRAMES].copy()
        stage=raw/'fit_input/external';stage.mkdir(parents=True)
        staged_meta=dict(metadata,samples=[metadata['samples'][i] for i in FRAMES])
        save(stage/'camera.json',staged_meta);np.savez_compressed(stage/'depth.npz',depth_raw=training_depth)
        r['initial_state']={'train_episode':EPISODES[0],'heldout_episodes':list(EPISODES[1:]),'frames_per_episode':FRAMES,
            'baseline_calibration_sha256':digest(basepath),'fitter_sha256':digest(fitter)}
        r['actions']=[{'method':'existing RANSAC/SVD plane fitter; no source modifications',
            'bounds':{'max_normal_change_deg':2.,'max_height_change_m':.02},
            'acceptance':'both heldout recordings RMSE improve >=10%, p95 does not increase; bounds and MuJoCo load check pass',
            'evaluation_selection':'same fixed ROIs and baseline-plane distance <20 mm for old/new'}]
        with (raw/'fit.log').open('w') as log:
            subprocess.run([sys.executable,str(fitter),'--capture',str(stage.parent)],cwd=SIM,
                           stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
        fitted=json.loads((stage.parent/'table_registration.json').read_text())
        candidate=np.array(fitted['T_world_rgb_estimate'])
        angle=float(np.rad2deg(np.arccos(np.clip(old[2,:3]@candidate[2,:3],-1,1))))
        dz=float(candidate[2,3]-old[2,3])
        bounds=angle<=2 and abs(dz)<=.02 and np.allclose(old[:2,3],candidate[:2,3])
        evaluations=[];evidence=[]
        for index,name in enumerate(EPISODES):
            p=DATA/'complete_real/failed'/name/'raw/task/external'
            meta=json.loads((p/'camera.json').read_text())
            if meta['serial']!=metadata['serial']:raise ValueError('Camera identity differs across episodes')
            with np.load(p/'depth.npz') as a:depths=a['depth_raw'][FRAMES].copy()
            errors=[[],[]]
            for frame,depth in zip(FRAMES,depths):
                cloud=measured_cloud(meta,depth);baseline=plane_error(cloud,old)
                select=abs(baseline)<.02
                errors[0].extend(baseline[select]);errors[1].extend(plane_error(cloud[select],candidate))
            evaluations.append({'source_episode':name,'role':'train' if index==0 else 'heldout',
                                'baseline':metrics(np.array(errors[0])),'candidate':metrics(np.array(errors[1]))})
            evidence.extend({'dataset_relative_path':str(f.relative_to(DATA)),'sha256':digest(f)} for f in (p/'camera.json',p/'depth.npz'))
        c=deepcopy(calibration);c['cameras']['overhead']['T_parent_rgb']=candidate.tolist()
        c['cameras']['overhead']['status']='candidate table-normal/distance fit only; no full hand-eye or contact calibration'
        save(raw/'candidate_cameras.json',c)
        # Check the actual existing MuJoCo loading interface, not merely the candidate JSON.
        sys.path.insert(0,str(SIM/'src'))
        from piper_sim.config import load
        from piper_sim.backends.mujoco import MujocoBackend
        task=yaml.safe_load((SIM/'configs/tasks/real_connector_calibrated.yaml').read_text())
        task['camera_calibration']=str(raw/'candidate_cameras.json')
        sim=MujocoBackend(dict(load(),task_scene=task))
        try:
            camera=sim.model.camera('overhead')
            import mujoco
            rot=np.empty(9);mujoco.mju_quat2Mat(rot,camera.quat)
            load_ok=bool(np.allclose(camera.pos,candidate[:3,3]) and
                         np.allclose(rot.reshape(3,3)@np.diag([1.,-1.,-1.]),candidate[:3,:3]))
        finally:sim.close()
        accepted=bool(bounds and load_ok and all(x['candidate']['rmse_mm']<=.9*x['baseline']['rmse_mm'] and
                     x['candidate']['p95_abs_mm']<=x['baseline']['p95_abs_mm'] for x in evaluations[1:]))
        r.update(result='success',task_success=None,validation_status='INDEPENDENT_GEOMETRY_VERIFIED' if accepted else 'CALIBRATION_REJECTED',
            expected_effects=['Improve measured camera/table geometry on separate recordings without changing unobservable XY/yaw or contact physics'],
            observed_effects=evaluations,repair={'accepted':accepted,'normal_change_deg':angle,'height_change_m':dz,
                'within_bounds':bool(bounds),'mujoco_load_verified':load_ok,'previous_T_world_rgb':old.tolist(),
                'candidate_T_world_rgb':candidate.tolist(),'active_default_changed':False},evidence=evidence,
            root_cause='UNKNOWN: independent plane residual does not identify real insertion failure cause',
            lessons_learned=['Only three camera/table degrees of freedom are observable from a plane; world XY/yaw remain CAD assumptions'],
            next_improvement='Use only validated observable geometry; task-cost benefits require separate simulation tests')
        r['limitations']=['Three frames per episode, same physical camera and workstation; not cross-installation validation',
            'RGB-D systematic bias persists across episodes; no external metrology',
            'Table world z=25 mm and CAD world XY/yaw remain fixed assumptions',
            'No calibrated aperture, contact force, friction, mass or COM',
            'No real motion or new task rollout; camera load check does not measure task success']
    except Exception as exc:r['error']=str(exc)
    finally:
        r['analysis_version']=digest(__file__);r['tokens']={'planner':0,'research_orchestrator':None}
        r['execution_time']=time.monotonic()-started;r['simulation_seconds']=0;finish(out,r)


if __name__=='__main__':main()

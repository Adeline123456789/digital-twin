from pathlib import Path
import json,sys,csv
import numpy as np,yaml,cv2,mujoco
from PIL import Image
S=Path('/media/midea/F496AC8096AC4548/piper_simulation');sys.path.insert(0,str(S/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
B=Path('/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug/self_evolution/spatial_cycle6');cap=B/'real_20261116T062116Z';views=json.loads((cap/'probe_views.json').read_text())['views'];rows=list(csv.DictReader((cap/'commanded_probe.csv').open()));t=np.array([float(r['monotonic']) for r in rows]);view=views[0];q=np.array([np.interp(view['exposure_monotonic_s'],t,[float(r[f'feedback_j{i}']) for r in rows]) for i in range(1,7)]);w=np.interp(view['exposure_monotonic_s'],t,[float(r['feedback_gripper_open'])*.095 for r in rows])
cfg=dict(load(),task_scene=yaml.safe_load((B/'probe_002/task.yaml').read_text()));sim=MujocoBackend(cfg);sim.set_observation_state(np.r_[np.deg2rad(q),w]);cid=sim.model.camera('right_wrist').id
base=json.loads((B/'fit_001/result.json').read_text())['baseline_T_link6_rgb'];out=B/'mount_visual_audit';out.mkdir(exist_ok=False);results={};real=cv2.imread(view['image_path']);gray=cv2.cvtColor(real,cv2.COLOR_BGR2GRAY);roi=np.zeros(gray.shape,bool);roi[330:450,:200]=True;roi[330:450,440:]=True
for name,mount in [('baseline',base)]+[(n,json.loads((B/n/'result.json').read_text())['T_link6_rgb']) for n in ['fit_001','fit_002']]:
 M=np.array(mount);sim.model.cam_pos[cid]=M[:3,3];mujoco.mju_mat2Quat(sim.model.cam_quat[cid],(M[:3,:3]@np.diag([1,-1,-1])).ravel());mujoco.mj_forward(sim.model,sim.data)
 Image.fromarray(sim._render('right_wrist',640,480)).save(out/(name+'.png'));renderer=sim.renderers[(640,480)];renderer.enable_segmentation_rendering();renderer.update_scene(sim.data,camera='right_wrist');seg=renderer.render().copy();renderer.disable_segmentation_rendering()
 ids=[i for i in range(sim.model.ngeom) if sim.model.body(sim.model.geom_bodyid[i]).name in ['right/link7','right/link8']];mask=np.isin(seg[:,:,0],ids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&roi
 metrics={}
 for threshold in [40,60,80]:
  ref=(gray<threshold)&roi
  # Restricted to the known fixed-finger regions, not the task object.
  intersection=np.logical_and(ref,mask).sum();union=np.logical_or(ref,mask).sum()
  metrics[str(threshold)]={'iou':float(intersection/union),'real_pixels':int(ref.sum()),'sim_pixels':int(mask.sum())}
 results[name]=metrics
result={'method':'Static rendering at measured holder-center joint state/aperture; only fixed finger image regions y330:450 x<200 orx>=440. Real gray thresholds40/60/80; simulated finger geometry masks use privileged segmentation. No optimizer or calibration fit.','metric_scope':'CAD/finger appearance and camera model joint check, not physical metrology or task success','results':results,'measured_aperture_m':w,'robot_motion_commands':0,'simulation_physics_steps':0}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));sim.close()

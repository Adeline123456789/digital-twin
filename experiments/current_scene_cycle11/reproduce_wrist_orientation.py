from pathlib import Path
import sys,json,time,numpy as np,yaml,cv2,mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path.cwd()/'scripts'))
from learn import DATA,save,digest
from plan_board_views import MujocoBackend,load
B=DATA/'self_evolution/end_to_end_cycle10';C=B/'current_comparison_001';O=DATA/'self_evolution/current_scene_cycle11/orientation_fit_001';O.mkdir(parents=True,exist_ok=False)
V=json.loads((C/'paired_views.json').read_text())['views'];s=MujocoBackend(dict(load(),task_scene=yaml.safe_load((B/'current_location_probe_002/task.yaml').read_text())))
start=time.monotonic();history=[];renderer=None
try:
 cid=s.model.camera('right_wrist').id;oldpos=s.model.cam_pos[cid].copy();rot=np.empty(9);mujoco.mju_quat2Mat(rot,s.model.cam_quat[cid]);oldrot=rot.reshape(3,3)@np.diag([1,-1,-1]);s.model.vis.quality.offsamples=0
 ids=[i for i in range(s.model.ngeom) if s.model.body(s.model.geom_bodyid[i]).name in ['right/link7','right/link8']];assert ids
 yy,xx=np.indices((480,640));roi=(yy>=280)&((xx<240)|(xx>440));real=[];states=[]
 for v in V:
  im=cv2.imread(str(C/(v['phase']+'_right_wrist.jpg')));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);real.append(gray<45)
  q=np.array(v['images']['right_wrist']['q_deg_width_m']);q[:6]=np.deg2rad(q[:6]);s.set_observation_state(q);d=mujoco.MjData(s.model);d.qpos[:]=s.data.qpos;mujoco.mj_forward(s.model,d);states.append(d)
 renderer=mujoco.Renderer(s.model,height=480,width=640);renderer.enable_segmentation_rendering()
 def render(x,i):
  r=oldrot@Rotation.from_euler('xz',x,degrees=True).as_matrix();mujoco.mju_mat2Quat(s.model.cam_quat[cid],(r@np.diag([1,-1,-1])).ravel());mujoco.mj_camlight(s.model,states[i]);renderer.update_scene(states[i],camera='right_wrist');im=renderer.render();return np.isin(im[:,:,0],ids)&(im[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
 def metric(pred,true):
  a=pred&roi;b=true&roi;return float((a&b).sum()/(a|b).sum())
 def cost(x):
  v=1-metric(render(x,0),real[0]);history.append([*map(float,x),v]);return v
 fit=minimize(cost,[0.,0.],method='Powell',bounds=[(-8,8),(-8,8)],options=dict(maxfev=140,xtol=.015,ftol=1e-5))
 rows=[]
 for i,v in enumerate(V):
  row=dict(phase=v['phase'],split='train' if i==0 else 'heldout',baseline_iou=metric(render([0,0],i),real[i]),candidate_iou=metric(render(fit.x,i),real[i]));rows.append(row)
  cv2.imwrite(str(O/(v['phase']+'_candidate_mask.png')),render(fit.x,i).astype('uint8')*255)
 m=np.eye(4);m[:3,3]=oldpos;m[:3,:3]=oldrot@Rotation.from_euler('xz',fit.x,degrees=True).as_matrix();save(O/'private_mount.json',dict(T_parent_rgb=m.tolist()))
 result=dict(candidate='wrist_optical_xz_rotation_only',delta_deg=fit.x.tolist(),translation_changed=False,intrinsics_changed=False,optimizer_success=bool(fit.success),nfev=fit.nfev,comparisons=rows,source_episode='20261116_real_current_location_000002',hardware_commands=0,calibration_deployed=False,scope='Current finger-CAD silhouette alignment only; not independent metric calibration',seconds=time.monotonic()-start,program_sha256=digest(__file__),private_mount_sha256=digest(O/'private_mount.json'))
 save(O/'result.json',result);save(O/'history.json',history);print(json.dumps(result))
finally:
 if renderer:renderer.close()
 s.close()

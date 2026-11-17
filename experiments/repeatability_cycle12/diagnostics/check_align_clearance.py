import sys,json,yaml,numpy as np,mujoco
from pathlib import Path
sys.path.insert(0,'scripts');from learn import DATA,SIM
sys.path.insert(0,str(SIM/'scripts'));from try_plug_pick_sim import PoseIK,MujocoBackend,load
b=DATA/'self_evolution/repeatability_cycle12';t=DATA/'self_evolution/20261116T125354Z_b5546f36';s=MujocoBackend(dict(load(),task_scene=yaml.safe_load((t/'task.yaml').read_text())))
try:
 with np.load(t/'final_state.npz') as d:s.data.qpos[:]=d['qpos'];s.data.qvel[:]=d['qvel'];s.data.ctrl[:]=d['ctrl']
 mujoco.mj_forward(s.model,s.data)
 def T(name):
  v=s.data.body(name);r=np.eye(4);r[:3,:3]=v.xmat.reshape(3,3);r[:3,3]=v.xpos;return r
 rel=np.linalg.inv(T('right/gripper_center'))@T('task/yellow_plug');rows=[]
 for gap in [.025,.024,.023,.022,.020]:
  desired=np.eye(4);desired[:2,3]=s.task.outlets[s.task.config['target_index']];desired[2,3]=s.task.socket_face_z+.0002+gap;g=T('task/strip')@desired@np.linalg.inv(rel);ik=PoseIK(s);ik.seed=s.data.qpos[s.indices['right']['qpos'][:6]].copy()
  try:q=ik.solve(g[:3,3],g[:3,:3]);row=dict(gap_mm=gap*1000,ik_pass=True,joints_deg=np.rad2deg(q).tolist())
  except ValueError as e:row=dict(gap_mm=gap*1000,ik_pass=False,error=str(e))
  rows.append(row)
 result=dict(source_trial=t.name,tests=rows,hardware_commands=0,privileged_information_used=True,scope='Static native IK at observed simulated carry state; not collision/dynamic validation')
 (b/'align_clearance_screen.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:s.close()

from pathlib import Path
import sys,json,yaml,numpy as np,mujoco
SIM=Path('/media/midea/F496AC8096AC4548/piper_simulation');sys.path.insert(0,str(SIM/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.mixed_recording import configure_rigid_collection
import trimesh
b=Path('/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug');t=b/'self_evolution/20261116T114608Z_4d3972ba'
rows=[json.loads(l) for l in (t/'trajectory.jsonl').open()];r=[r for r in rows if r['phase']=='grasp'][-1]
s=MujocoBackend(dict(load(),task_scene=yaml.safe_load((t/'task.yaml').read_text())));configure_rigid_collection(s);s.set_observation_state(np.array(r['q']))
bid=s.model.body('task/yellow_plug').id;jid=s.model.body_jntadr[bid];adr=s.model.jnt_qposadr[jid];s.data.qpos[adr:adr+7]=r['plug']+r['plug_quaternion_wxyz'];mujoco.mj_forward(s.model,s.data)
opts=mujoco.MjvOption();summary=[]
for c in s.data.contact:
 names=[s.model.geom(g).name for g in [c.geom1,c.geom2]]
 if not any(n in ['gripper_right_46','gripper_right_36'] for n in names):continue
 item={'pair':names,'penetration_mm':-c.dist*1000,'surfaces':[]}
 for g in [c.geom1,c.geom2]:
  body=s.model.geom_bodyid[g];nearest=[]
  for v in np.flatnonzero(s.model.geom_bodyid==body):
   if not opts.geomgroup[s.model.geom_group[v]] or s.model.geom_rgba[v,3]==0 or s.model.geom_type[v]!=mujoco.mjtGeom.mjGEOM_MESH:continue
   mid=s.model.geom_dataid[v];vs=s.model.mesh_vertadr[mid];fs=s.model.mesh_faceadr[mid]
   verts=s.model.mesh_vert[vs:vs+s.model.mesh_vertnum[mid]];faces=s.model.mesh_face[fs:fs+s.model.mesh_facenum[mid]]
   q=(c.pos-s.data.geom_xpos[v])@s.data.geom_xmat[v].reshape(3,3)
   _,distance,_=trimesh.proximity.closest_point_naive(trimesh.Trimesh(vertices=verts,faces=faces,process=False),q[None,:])
   nearest.append((float(distance[0])*1000,s.model.geom(v).name))
  item['surfaces'].append({'collision_geom':s.model.geom(g).name,'geom_type':int(s.model.geom_type[g]),'geom_group':int(s.model.geom_group[g]),'rgba':s.model.geom_rgba[g].tolist(),'nearest_visible_mesh_distance_mm':sorted(nearest)[:2]})
 summary.append(item)
s.apply(s.layout.command(np.array(r["action"]),s.sim_time,0));mujoco.mj_forward(s.model,s.data)
print("ACTUATORS",json.dumps([dict(name=s.model.actuator(i).name,force=float(s.data.actuator_force[i]),forcerange=s.model.actuator_forcerange[i].tolist(),forcelimited=bool(s.model.actuator_forcelimited[i]),gainprm=s.model.actuator_gainprm[i,:3].tolist(),ctrlrange=s.model.actuator_ctrlrange[i].tolist(),gear=s.model.actuator_gear[i,:3].tolist()) for i in range(s.model.nu)]))
s.close();print(json.dumps(summary,indent=2))
(b/'self_evolution/repeatability_cycle12/grasp_surface_check.json').write_text(json.dumps(summary,indent=2)+'\n')

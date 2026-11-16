"""Plan noncontact board views with existing native IK; no hardware interface."""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from learn import SIM,save,digest
from chessboard_observation import board_points
sys.path[:0]=[str(SIM/'src'),str(SIM/'scripts')]
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from try_plug_pick_sim import PoseIK


def body_pose(sim,name):
    b=sim.data.body(name);t=np.eye(4);t[:3,:3]=b.xmat.reshape(3,3);t[:3,3]=b.xpos
    return t


def camera_pose(sim,name):
    i=sim.model.camera(name).id;t=np.eye(4)
    t[:3,:3]=sim.data.cam_xmat[i].reshape(3,3)@np.diag([1,-1,-1]);t[:3,3]=sim.data.cam_xpos[i]
    return t


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    sim=MujocoBackend(dict(load(),task_scene=yaml.safe_load(a.task.read_text())))
    try:
        obs=json.loads(a.observation.read_text());candidate=obs['candidates'][0]
        board=camera_pose(sim,'overhead')@np.array(candidate['T_camera_target'])
        mount=np.linalg.inv(body_pose(sim,'right/gripper_center'))@camera_pose(sim,'right_wrist')
        points=board_points()@board[:3,:3].T+board[:3,3]
        K=np.array(json.loads(a.camera.read_text())['K'])
        ik=PoseIK(sim);results=[];failures=0
        for dx in [-.14,-.07,.0,.07]:
            for dy in [-.14,-.07,.0]:
                for height in [.40,.44]:
                    cam=np.eye(4);cam[:3,3]=board[:3,3]+[dx,dy,height]
                    forward=board[:3,3]-cam[:3,3];forward/=np.linalg.norm(forward)
                    right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
                    cam[:3,:3]=np.column_stack([right,np.cross(forward,right),forward])
                    goal=cam@np.linalg.inv(mount)
                    ik.seed=np.deg2rad([-8,76,-57,0,55,-104])
                    try:q=ik.solve(goal[:3,3],goal[:3,:3])
                    except ValueError:failures+=1;continue
                    sim.set_observation_state(np.r_[q,.060])
                    actual=camera_pose(sim,'right_wrist');xyz=(points-actual[:3,3])@actual[:3,:3]
                    uv=xyz@K.T;uv=uv[:,:2]/uv[:,2:]
                    margin=float(min(uv[:,0].min(),640-uv[:,0].max(),uv[:,1].min(),480-uv[:,1].max()))
                    if xyz[:,2].min()<=0 or margin<35:continue
                    results.append(dict(q_deg=np.rad2deg(q).tolist(),camera_offset_m=[dx,dy,height],
                        corner_margin_px=margin,gripper_world_m=goal[:3,3].tolist()))
        save(a.output/'private_scene_estimate.json',dict(board_world_pose=board.tolist(),
            gripper_camera_pose=mount.tolist(),board_pose_scope='Provisional existing external registration; observation planning only',
            input_sha256={p.name:digest(p) for p in [a.observation,a.task,a.camera]}))
        save(a.output/'candidates.json',dict(candidates=results,ik_failures=failures,proposed=24,
            motion_commands=0,physical_path_validated=False))
        print(json.dumps(dict(feasible_views=len(results),ik_failures=failures,
            board_center_height_m=float(board[2,3]),candidates=[{k:v for k,v in r.items() if k!='gripper_world_m'} for r in results])))
    finally:sim.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['task','observation','camera','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())

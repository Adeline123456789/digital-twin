"""Replay frozen controller commands in existing MuJoCo, keeping its integrator/contact gates."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import yaml
from learn import SIM, save, digest
from trajectory_contract import load_reference
from contact_evidence import grasp_relation
sys.path.insert(0,str(SIM/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.mixed_recording import configure_rigid_collection,collision_snapshot


def validate(trial,reference):
    trial,reference=Path(trial),Path(reference)
    manifest=json.loads((reference/'manifest.json').read_text())
    if digest(trial/'trajectory.jsonl')!=manifest['source_trajectory_sha256']:
        raise ValueError('Reference source changed')
    timeline,targets,phases=load_reference(reference)
    config=yaml.safe_load((trial/'task.yaml').read_text())
    sim=MujocoBackend(dict(load(),task_scene=config))
    started=time.monotonic();worst=0.;history=[];last_time=0.;result={
        'reference_sha256':manifest['trajectory_id'],'result':'infrastructure_error',
        'task_success':False,'all_substeps_valid':False,'real_commands':0,
        'privileged_information_used':True,'physics_parameter_changes':False,
        'scope':'100Hz frozen target commands; original numerical integrator, contact bounds and task success detector'}
    try:
        result['profile']=configure_rigid_collection(sim)
        with np.load(trial/'initial_state.npz') as initial:
            sim.data.qpos[:]=initial['qpos'];sim.data.qvel[:]=initial['qvel'];sim.data.ctrl[:]=initial['ctrl']
        mujoco.mj_forward(sim.model,sim.data)
        if sim.arms!=('right',):raise ValueError('This pair uses only the right arm')
        joints=[sim.layout.names.index(f'right.joint{i}') for i in range(1,7)]
        grip=sim.layout.names.index('right.gripper_width')
        index=-1;next_observation=1/30
        while sim.sim_time < timeline[-1]+.01:
            current=min(np.searchsorted(timeline,sim.sim_time,side='right')-1,len(timeline)-1)
            if current!=index:
                index=current;q=sim.control_targets();q[joints]=np.deg2rad(targets[index,:6]);q[grip]=targets[index,6]*.095
                sim.apply(sim.layout.command(q,sim.sim_time,index))
            mujoco.mj_step(sim.model,sim.data);mujoco.mj_forward(sim.model,sim.data)
            penetration=max(0.,-min((c.dist for c in sim.data.contact),default=0.))
            worst=max(worst,penetration)
            if penetration>.0002:
                result.update(result='failure',first_anomaly={'time_s':sim.sim_time,'phase':phases[index],'reason':'Original penetration bound exceeded','contacts':collision_snapshot(sim.model,sim.data)})
                break
            if sim.sim_time+1e-9>=next_observation:
                sim.task.metrics(sim.task_adapter.task_poses(sim.data,sim.task),sim.sim_time-last_time)
                last_time=sim.sim_time;next_observation+=1/30
                obs=sim.read()
                history.append(dict(time=sim.sim_time,reference_index=int(index),phase=phases[index],q=obs.position.tolist(),
                    action=q.tolist(),plug=sim.data.body('task/yellow_plug').xpos.tolist(),
                    grasp_relation=grasp_relation(sim.data),
                    gripper=sim.data.body('right/gripper_center').xpos.tolist(),penetration_m=penetration,
                    task_metrics=obs.extras['task']['metrics']))
        else:
            success=bool(sim.read().extras['task']['metrics']['success'])
            result.update(result='success' if success else 'failure',task_success=success,all_substeps_valid=True)
        result.update(max_penetration_m=worst,simulated_seconds=sim.sim_time,observation_frames=len(history),
            applied_reference_steps=int(index+1),integrator_timestep_s=float(sim.model.opt.timestep))
    except Exception as exc:result['error']=str(exc)
    finally:
        sim.close();result['wall_seconds']=time.monotonic()-started
        trace=reference/'prediction.jsonl';trace.write_text(''.join(json.dumps(r)+'\n' for r in history))
        result['prediction_sha256']=digest(trace)
        save(reference/'validation.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',required=True,type=Path);p.add_argument('--reference',required=True,type=Path);a=p.parse_args()
    print(json.dumps(validate(a.trial,a.reference),indent=2))

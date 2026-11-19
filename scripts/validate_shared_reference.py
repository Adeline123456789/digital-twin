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


def upright_stage_check(history, limit_deg=1.0):
    rows = [r for r in history if r['phase'] in ('lift', 'reorient', 'transport')]
    angles = [float(np.rad2deg(r['task_metrics']['tilt_rad'])) for r in rows]
    return dict(passed=bool(angles) and all(np.isfinite(a) and a <= limit_deg for a in angles),
                max_tilt_deg=max(angles) if angles else None, limit_deg=limit_deg,
                phases=sorted({r['phase'] for r in rows}),
                scope='Simulated absolute plug axis during lift/reorient/transport; fold excluded; not a real angle measurement')


def validate(trial,reference):
    trial,reference=Path(trial),Path(reference)
    manifest=json.loads((reference/'manifest.json').read_text())
    if digest(trial/'trajectory.jsonl')!=manifest['source_trajectory_sha256']:
        raise ValueError('Reference source changed')
    diagnostic = manifest.get('task_scope') == 'precontact_simulation_only'
    if diagnostic:
        from task_relative import pin_error, forbidden_contact, pin_points, grasp_drift
        goal = np.asarray(manifest['T_SP_desired'])
        if (digest(trial/'initial_state.npz') != manifest['source_initial_state_sha256']
                or digest(trial/'task.yaml') != manifest['task_sha256']):
            raise ValueError('Precontact initial state or scene changed')
    timeline,targets,phases=load_reference(reference)
    if diagnostic and set(phases)-{'approach','descend','grasp','preload','lift','reorient','transport','align'}:
        raise ValueError('Contact phase in precontact reference')
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
        if diagnostic and manifest.get('initial_strip_sha256'):
            strip = trial/'initial_strip.npz'
            if digest(strip) != manifest['initial_strip_sha256']:
                raise ValueError('Measured socket initialization changed')
            with np.load(strip) as initial:
                body=sim.model.body('task/strip').id
                sim.model.body_pos[body]=initial['pos']
                sim.model.body_quat[body]=initial['quat']
        if diagnostic and manifest.get('initial_integration_sha256'):
            source = trial/'initial_integration.json'
            if digest(source) != manifest['initial_integration_sha256']:
                raise ValueError('Changed held control')
            state=json.loads(source.read_text())
            mujoco.mj_setState(sim.model,sim.data,np.array(state['values']),state['spec'])
        mujoco.mj_forward(sim.model,sim.data)
        if sim.arms!=('right',):raise ValueError('This pair uses only the right arm')
        joints=[sim.layout.names.index(f'right.joint{i}') for i in range(1,7)]
        grip=sim.layout.names.index('right.gripper_width')
        if diagnostic:
            points=pin_points(sim)
            start_grasp=manifest.get('T_GP_start')
            if start_grasp is None:
                observed=json.loads((trial/'relative_result.json').read_text())['events']
                start_grasp=next(e['observation']['T_GP'] for e in observed if e['event']=='OBSERVE_TARGET')
            g0=np.asarray(start_grasp)
            max_grasp_drift=0.
        index=-1;next_observation=1/30
        while sim.sim_time < timeline[-1]+.01:
            current=min(np.searchsorted(timeline,sim.sim_time,side='right')-1,len(timeline)-1)
            if current!=index:
                index=current;q=sim.control_targets();q[joints]=np.deg2rad(targets[index,:6]);q[grip]=targets[index,6]*.095
                sim.apply(sim.layout.command(q,sim.sim_time,index))
            mujoco.mj_step(sim.model,sim.data);mujoco.mj_forward(sim.model,sim.data)
            penetration=max(0.,-min((c.dist for c in sim.data.contact),default=0.))
            worst=max(worst,penetration)
            if diagnostic:
                contact = forbidden_contact(collision_snapshot(sim.model,sim.data), True)
                if contact:
                    result.update(result='failure', first_anomaly=dict(time_s=sim.sim_time,
                        phase=phases[index],reason='Forbidden contact in precontact replay',contact=contact))
                    break
            if penetration>.0002:
                result.update(result='failure',first_anomaly={'time_s':sim.sim_time,'phase':phases[index],'reason':'Original penetration bound exceeded','contacts':collision_snapshot(sim.model,sim.data)})
                break
            if sim.sim_time+1e-9>=next_observation:
                sim.task.metrics(sim.task_adapter.task_poses(sim.data,sim.task),sim.sim_time-last_time)
                last_time=sim.sim_time;next_observation+=1/30
                if diagnostic and phases[index] in ('transport','align'):
                    max_grasp_drift=max(max_grasp_drift,grasp_drift(sim,g0,points))
                obs=sim.read()
                history.append(dict(time=sim.sim_time,reference_index=int(index),phase=phases[index],q=obs.position.tolist(),
                    action=q.tolist(),plug=sim.data.body('task/yellow_plug').xpos.tolist(),
                    grasp_relation=grasp_relation(sim.data),
                    gripper=sim.data.body('right/gripper_center').xpos.tolist(),penetration_m=penetration,
                    task_metrics=obs.extras['task']['metrics']))
        else:
            success=not diagnostic and bool(sim.read().extras['task']['metrics']['success'])
            result.update(result='success' if success else 'failure',task_success=success,all_substeps_valid=True)
        if diagnostic:
            error=pin_error(sim,goal)
            upright = upright_stage_check(history)
            passed = (result['all_substeps_valid'] and max(error.values()) <= sim.task.config['slot_clearance_m']
                      and max_grasp_drift <= sim.task.config['slot_clearance_m']
                      and (manifest['held_control'] or upright['passed']))
            result.update(result='PRECONTACT_VALIDATED' if passed else 'PRECONTACT_REJECTED',
                          precontact_error=error,max_grasp_drift_m=max_grasp_drift,task_success=False,stage_success=passed)
        result.update(upright_stage=upright_stage_check(history),
            max_penetration_m=worst,simulated_seconds=sim.sim_time,observation_frames=len(history),
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

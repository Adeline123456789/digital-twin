"""Offline native-IK probes at saved transport failures; no simulation steps or hardware."""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from learn import DATA, ROOT, SIM, digest, finish, record
from episode_format import source_for

sys.path.insert(0,str(SIM/'scripts'))
from try_plug_pick_sim import PoseIK, MujocoBackend, load


def body_transform(sim, name):
    body=sim.data.body(name)
    transform=np.eye(4)
    transform[:3,:3]=body.xmat.reshape(3,3)
    transform[:3,3]=body.xpos
    return transform


def probe(ident, gaps):
    out,result=record('research_stage4','Native IK approach-height probes for '+ident,'offline_kinematic_simulation')
    started=time.monotonic()
    sim=None
    try:
        source_report=next((ROOT/'experiments').glob('*/'+ident+'/result.json'))
        source_record=json.loads(source_report.read_text())
        assert source_for(source_record)==SIM, 'Run with the source profile that produced this Episode'
        native_report=source_record['observed_effects'][0]
        assert native_report['last_phase']=='transport' and 'Unreachable pose' in native_report['error']
        raw=DATA/'self_evolution'/ident
        cfg=yaml.safe_load((raw/'task.yaml').read_text())
        trace=[json.loads(x) for x in (raw/'trajectory.jsonl').read_text().splitlines()]
        assert trace[-1]['phase']=='lift', 'Probe must start before failed transport, not after contact'
        sim=MujocoBackend(dict(load(),task_scene=cfg))
        with np.load(raw/'final_state.npz') as state:
            for key in ('qpos','qvel','ctrl'):getattr(sim.data,key)[:]=state[key]
        mujoco.mj_forward(sim.model,sim.data)
        relative=np.linalg.inv(body_transform(sim,'right/gripper_center'))@body_transform(sim,'task/yellow_plug')
        socket=body_transform(sim,'task/strip')
        ik=PoseIK(sim)
        native_seed=np.array(trace[-1]['action'][7:13])
        attempts=[]
        assert gaps[0]==native_report['preinsert_gap_m'], 'First probe must reproduce the saved baseline'
        native_goal=np.fromstring(native_report['error'].split('pose [')[1].split(']')[0],sep=' ')
        for gap in gaps:
            desired=np.eye(4)
            desired[:2,3]=sim.task.outlets[cfg['target_index']]
            desired[:2,3]+=np.array(native_report['deliberate_insertion_offset_mm'])/1000
            desired[2,3]=sim.task.socket_face_z+.0002+gap
            if gap>.03:desired[:3,:3]=Rotation.from_euler('x',5,degrees=True).as_matrix()
            target=socket@desired@np.linalg.inv(relative)
            if not attempts:
                reproduction_error=float(np.max(abs(target[:3,3]-native_goal)))
                assert reproduction_error<=1e-8, 'Saved transport target not reproduced; do not search'
            ik.seed=native_seed.copy()
            t=time.monotonic();error=None
            try:
                ik.solve(target[:3,3],target[:3,:3]);ok=True
            except ValueError as exc:
                ok=False;error=str(exc)
            position_error=float(np.linalg.norm(ik.data.xpos[ik.bid]-target[:3,3]))
            angle_error=float(np.linalg.norm(Rotation.from_matrix(target[:3,:3].T@ik.data.xmat[ik.bid].reshape(3,3)).as_rotvec()))
            attempts.append(dict(preinsert_gap_m=gap,carry_tilt_deg=5 if gap>.03 else 0,
                ik_success=ok,error=error,target_world_xyz_m=target[:3,3].tolist(),
                position_error_m=position_error,orientation_error_deg=float(np.rad2deg(angle_error)),
                solver_seconds=time.monotonic()-t,joints_deg=np.rad2deg(ik.data.qpos[ik.ids]).tolist()))
        # An IK error is a solver result, not proof of global unreachability or task failure.
        result.update(result='success',task_success=None,validation_status='KINEMATIC_PROBE_ONLY',
            initial_state={'source_episode':ident,'target_index':cfg['target_index'],'phase':'end of lift'},
            baseline_goal_reproduction_max_error_m=reproduction_error,
            actions=[{'native_ik_gap_candidates_m':gaps}],experiments=attempts,
            observed_effects=[{'saved_goal_error':native_report['error']}],
            expected_effects=['Diagnose whether existing approach-height parameter changes native IK feasibility'],
            tokens=0,llm_inference_seconds=0,simulation_steps=0,
            privileged_information_used=True,privileged_fields_used=['saved simulator qpos/qvel/ctrl','plug/gripper/socket transforms','native IK joint state'],
            human_interventions=None,research_orchestrator_tokens=None,
            evidence=[{'dataset_relative_path':str((raw/name).relative_to(DATA)),'sha256':digest(raw/name)} for name in ['task.yaml','final_state.npz','trajectory.jsonl','report.json']],
            lessons_learned=['Native solver restored from recorded state; no full execution, collision-free path or seated success inferred from IK'],
            next_improvement='Verify baseline target reproduction, then execute any feasible candidate using unchanged native collision/success checks')
    except Exception as exc:
        result['error']=str(exc).replace(str(DATA),'$DATA').replace(str(ROOT),'$PROJECT_ROOT')
    finally:
        if sim is not None:sim.close()
        result['execution_time']=time.monotonic()-started;finish(out,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode');p.add_argument('--gaps',type=float,nargs='+',default=[.045,.04,.035,.031]);a=p.parse_args()
    if any(not .031<=gap<=.045 for gap in a.gaps):raise ValueError('This probe changes height only; keep existing five-degree carry tilt')
    probe(a.episode,a.gaps)

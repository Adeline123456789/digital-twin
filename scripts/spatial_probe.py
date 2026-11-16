"""Same-command, open-gripper spatial observation probe using the existing robot interfaces."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from learn import SIM, digest, save


def make_reference(entry, viewpoints):
    """Retain the native quintic12deg/s target profile; holds are for RGB measurement."""
    goals = []
    for view in viewpoints:
        q = np.r_[view['q_deg'], .060/.095]
        name = view['name']
        goals.append((name + '_center', q.copy()))
        for joint, delta in zip((3,4,5), view.get('joint_variation_deg',[6.,-6.,6.])):
            varied = q.copy(); varied[joint] += delta
            goals.extend([(name + f'_j{joint+1}', varied), (name + f'_return{joint+1}', q.copy())])
        if 'translation_joint_deg' in view:
            goals.extend([(name + '_translation', np.r_[view['translation_joint_deg'], .060/.095]),
                          (name + '_translation_return', q.copy())])
        varied = q.copy(); varied[[3, 4, 5]] += view.get('heldout_variation_deg',[-4,-3,-4])
        goals.extend([(name + '_heldout', varied), (name + '_return_heldout', q.copy())])
    goals.append(('holder_return', np.r_[viewpoints[0]['q_deg'], .060/.095]))
    goals.append(('fold', np.r_[np.zeros(6), .060/.095]))
    q = np.r_[entry, .060/.095]
    targets, phases = [q.copy()], ['entry']
    for name, goal in goals:
        seconds = max(.35, 1.875*np.max(abs(goal[:6]-q[:6]))/12.)
        count = int(np.ceil(seconds*100))
        for a in np.arange(1, count+1)/count:
            targets.append(q+(goal-q)*(a**3*(10-15*a+6*a*a))); phases.append(name)
        targets.extend([goal.copy() for _ in range(80)])
        phases.extend([name + '_hold']*80)
        q = goal
    return np.arange(len(targets))/100, np.array(targets), phases


def add_observation_obstacle(sim, obstacle, output):
    """Add a local conservative board proxy, colliding only with the robot."""
    import mujoco
    from arm_layout import save_recorded_model
    from scipy.spatial.transform import Rotation
    pose=np.array(obstacle['world_pose']);size=np.array(obstacle['half_size_m'])
    if pose.shape!=(4,4) or size.shape!=(3,) or not np.isfinite(pose).all() or np.any(size<=0):
        raise ValueError('Invalid observation obstacle')
    if np.any((sim.model.geom_contype|sim.model.geom_conaffinity)&4):
        raise ValueError('Observation collision bit already used by the native model')
    native=output/'native_model.xml';save_recorded_model(sim.model,native)
    spec=mujoco.MjSpec.from_file(str(native))
    quat=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
    body=spec.worldbody.add_body(name='task/observation_target',pos=pose[:3,3],quat=quat)
    body.add_geom(name='observation_target_proxy',type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=size,rgba=[.7,.4,.2,1],contype=4,conaffinity=0)
    old=sim.model;sim.model=spec.compile();sim.data=mujoco.MjData(sim.model)
    for field in ('body_mass','body_inertia','jnt_range','actuator_gainprm','actuator_biasprm'):
        before=getattr(old,field);after=getattr(sim.model,field)[:len(before)]
        if not np.allclose(before,after,rtol=1e-5,atol=1e-9):
            raise ValueError('Observation model changed native parameter: '+field)
    for side,ids in sim.indices.items():
        for i,name in enumerate([f'{side}/joint{j}' for j in range(1,9)]):
            if sim.model.joint(name).qposadr[0]!=ids['qpos'][i]:
                raise ValueError('Added static obstacle changed joint mapping')
    for i in range(sim.model.ngeom):
        if sim.model.body(sim.model.geom_bodyid[i]).name.startswith('right/'):
            sim.model.geom_conaffinity[i]|=4
    # Preserve runtime camera calibration after model serialization/compilation.
    for name in sim.rig['cameras']:
        before=old.camera(name).id;after=sim.model.camera(name).id
        for field in ('cam_pos','cam_quat','cam_intrinsic','cam_resolution'):
            getattr(sim.model,field)[after]=getattr(old,field)[before]
    sim.reset();save(output/'observation_obstacle.json',obstacle)


def build(task, viewpoints, entry, output, obstacle=None):
    import mujoco
    import yaml
    from PIL import Image
    sys.path[:0] = [str(SIM/'src')]
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.config import load
    from piper_sim.mixed_recording import configure_rigid_collection, collision_snapshot
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(task.read_text())
    config['robot_gravity_compensation'] = True
    sim = MujocoBackend(dict(load(), task_scene=config))
    if obstacle is not None:add_observation_obstacle(sim,obstacle,output)
    assert sim.arms == ('right',)
    timeline, targets, phases = make_reference(entry, viewpoints)
    np.savez(output/'reference.npz', time=timeline, targets=targets, phases=np.array(phases))
    save(output/'manifest.json', dict(purpose='noncontact_spatial_probe', trajectory_id=digest(output/'reference.npz'),
        task_sha256=digest(task), program_sha256=digest(__file__), joint_names=[f'right.joint{i}' for i in range(1,7)],
        joint_unit='degree', gripper_unit='width_m/.095', speed_limit_deg_s=12,
        initial_right_deg=entry, viewpoints=viewpoints, duration_s=float(timeline[-1]),
        observation_obstacle_sha256=digest(output/'observation_obstacle.json') if obstacle is not None else None,
        note='No grasp/insert task. Open-gripper observation holds and return; privileged simulation validation.'))
    (output/'task.yaml').write_text(yaml.safe_dump(config))
    result = dict(result='infrastructure_error', task_success=None, probe_valid=False,
                  reference_sha256=digest(output/'reference.npz'), privileged_information_used=True,
                  real_commands=0, all_substeps_valid=False, no_robot_task_contact=False)
    history = []; worst = 0.; started = time.monotonic(); last_index = -1; next_frame = 1/30
    try:
        result['profile'] = configure_rigid_collection(sim)
        q = np.r_[np.deg2rad(entry), .060]; sim.set_observation_state(q)
        np.savez(output/'initial_state.npz', qpos=sim.data.qpos, qvel=sim.data.qvel, ctrl=sim.data.ctrl)
        while sim.sim_time < timeline[-1]+.01:
            index = min(int(sim.sim_time*100+1e-8), len(targets)-1)
            if index != last_index:
                if last_index >= 0 and phases[last_index].endswith('_hold') and phases[index] != phases[last_index]:
                    for camera in ('overhead', 'right_wrist'):
                        Image.fromarray(sim._render(camera, 640, 480)).save(output/(phases[last_index]+'_'+camera+'.png'))
                q = np.r_[np.deg2rad(targets[index,:6]), targets[index,6]*.095]
                sim.apply(sim.layout.command(q, sim.sim_time, index)); last_index = index
            mujoco.mj_step(sim.model, sim.data); mujoco.mj_forward(sim.model, sim.data)
            worst = max(worst, -min((c.dist for c in sim.data.contact), default=0.))
            robot_task_contact = any(
                ((sim.model.body(sim.model.geom_bodyid[c.geom1]).name.startswith('right/') and
                  sim.model.body(sim.model.geom_bodyid[c.geom2]).name.startswith('task/')) or
                 (sim.model.body(sim.model.geom_bodyid[c.geom2]).name.startswith('right/') and
                  sim.model.body(sim.model.geom_bodyid[c.geom1]).name.startswith('task/')))
                for c in sim.data.contact if c.dist <= 0)
            if worst > .0002 or robot_task_contact:
                result.update(result='failure', failure_phase=phases[index], contacts=collision_snapshot(sim.model, sim.data))
                break
            if sim.sim_time+1e-9 >= next_frame:
                next_frame += 1/30
                history.append(dict(time_s=sim.sim_time, phase=phases[index], q=sim.read().position.tolist(),
                    gripper_xyz_m=sim.data.body('right/gripper_center').xpos.tolist(),
                    plug_xyz_m=sim.data.body('task/yellow_plug').xpos.tolist()))
        else:
            result.update(result='success', probe_valid=True, all_substeps_valid=True, no_robot_task_contact=True)
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        sim.close(); result.update(wall_seconds=time.monotonic()-started, max_penetration_m=worst)
        (output/'prediction.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
        result['prediction_sha256'] = digest(output/'prediction.jsonl')
        save(output/'validation.json', result)
    print(json.dumps(result, indent=2))


def load_probe(reference):
    manifest = json.loads((reference/'manifest.json').read_text())
    validation = json.loads((reference/'validation.json').read_text())
    identity = digest(reference/'reference.npz')
    if (manifest['purpose'] != 'noncontact_spatial_probe' or manifest['trajectory_id'] != identity or
        validation['reference_sha256'] != identity or not validation['probe_valid'] or
        not validation['all_substeps_valid'] or not validation['no_robot_task_contact']):
        raise ValueError('Frozen noncontact probe has not passed its physical validation')
    with np.load(reference/'reference.npz', allow_pickle=False) as data:
        return manifest, data['time'].copy(), data['targets'].copy(), data['phases'].tolist()


def require_fresh_rgb(capture):
    for camera in ('external','right_wrist'):
        frames=sorted((capture/camera).glob('*.jpg'))
        if len(frames)<2 or time.time()-frames[-2].stat().st_mtime>.5:
            raise RuntimeError('Fresh RGB lost; stop probe')


def execute(reference, capture):
    manifest, timeline, targets, phases = load_probe(reference)
    root = Path('/media/midea/F496AC8096AC4548/act_td3/piper_use_right')
    sys.path[:0] = [str(root)]
    from replay_connector import Hardware, Keyboard, parse_args, transition
    for camera in ('external', 'right_wrist'):
        if not (capture/camera/'READY').exists():raise RuntimeError('RGB not ready: '+camera)
    # Record before the controller's interactive/reset setup, which can exceed
    # the native capture program's90s trigger deadline. No motor is enabled here.
    (capture/'START').write_text(str(time.monotonic()))
    deadline=time.monotonic()+2
    while True:
        try:require_fresh_rgb(capture);break
        except RuntimeError:
            if time.monotonic()>=deadline:raise
            time.sleep(.03)
    args = parse_args([]); args.folded_entry = True; args.max_joint_speed = 12.; args.control_hz = 100
    hardware = Hardware(args, ['right'])
    report = dict(reference_id=manifest['trajectory_id'], purpose=manifest['purpose'],
                  completed_sequence=False, physical_task_success=None, real_depth=False,
                  execution_program_sha256=digest(__file__),
                  execution_mode='frozen_noncontact_probe', object_contact_intended=False)
    def before_enable(entry):
        if np.max(abs(entry[0,:6]-targets[0,:6])) > 1:raise ValueError('Probe start is stale')
        require_fresh_rgb(capture)
    try:
        entry = hardware.connect(SimpleNamespace(episode_id='SPATIAL_PROBE', actions={'right':targets}),
                                 targets[:1], confirmation='PROBE right', before_enable=before_enable)
        hardware.controllers['right'].joint_speed_percent = 15
        with (capture/'commanded_probe.csv').open('x', newline='') as f, Keyboard() as keyboard:
            writer = csv.writer(f)
            writer.writerow(['monotonic','phase']+[f'{k}_j{i}' for k in ('target','feedback') for i in range(1,7)]+
                            ['target_gripper_open','feedback_gripper_open'])
            last_phase = None; last_camera_check = 0.
            def emit(target, phase, duration):
                nonlocal last_phase, last_camera_check
                tick = time.monotonic(); hardware.poll(keyboard)
                if tick-last_camera_check > .2:
                    require_fresh_rgb(capture)
                    last_camera_check = tick
                if phase != last_phase:
                    print('PROBE_PHASE', phase, flush=True)
                    save(capture/'phase.json', dict(phase=phase, monotonic=tick)); last_phase=phase
                hardware.send(target); actual=hardware.feedback('right',True)[0]
                writer.writerow([time.monotonic(),phase,*target[0,:6],*actual[:6],target[0,6],actual[6]]); f.flush()
                time.sleep(max(0,duration-(time.monotonic()-tick)))
            for q in transition(entry,targets[:1],1.,args):emit(q,'prepare_gripper',.01)
            for i,(q,phase) in enumerate(zip(targets,phases)):
                emit(q[None,:],phase,timeline[i+1]-timeline[i] if i+1<len(timeline) else .01)
            hardware.wait_targets(targets[-1:],gripper=False,keyboard=keyboard)
            for _ in range(100):emit(targets[-1:],'final_hold',.01)
            report.update(completed_sequence=True, final_feedback=hardware.feedback('right',True)[0].tolist())
            print('PROBE_COMPLETE: folded, waiting for supported shutdown',flush=True)
    except BaseException as exc:
        report['error']=str(exc); raise
    finally:
        time.sleep(.15); (capture/'STOP').write_text('probe stopped\n'); save(capture/'execution_report.json',report)
        hardware.shutdown()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build'); b.add_argument('--task',type=Path,required=True); b.add_argument('--viewpoints',type=Path,required=True)
    b.add_argument('--entry',type=float,nargs=6,required=True); b.add_argument('--output',type=Path,required=True)
    b.add_argument('--obstacle',type=Path)
    e=sub.add_parser('execute'); e.add_argument('--reference',type=Path,required=True); e.add_argument('--capture',type=Path,required=True)
    e.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.mode=='build':build(a.task,json.loads(a.viewpoints.read_text()),a.entry,a.output,
                             json.loads(a.obstacle.read_text()) if a.obstacle else None)
    elif a.execute:execute(a.reference,a.capture)
    else:
        m,t,q,ph=load_probe(a.reference);print(json.dumps(dict(manifest=m,steps=len(q),hardware_commands=0),indent=2))

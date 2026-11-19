"""Right-arm precontact adapter. All physics and commands in this module are simulation-only."""
import importlib.util
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import yaml

from learn import DATA, ROOT, SIM, digest, save
from contact_evidence import integration_snapshot
from relative_pose import task_error

PRIVATE = DATA / 'self_evolution/right_relative_pose/round3_hook/try_plug_pick_sim.py'
BASE_HASH = '146934520588d4529e595a746e75a334c7394b6e5d00c3b4ab31cf178fa371ec'
HOOK_HASH = 'ebd74f3074f92083ebbae1c793239ab6550aeb73f399749bf3a2ef48baf4ba42'
HELD = '20261117T121435Z_bf6333a2'
REGRASP = '20261117T071708Z_94246e78'
SCENE = DATA / 'self_evolution/grasp_six_rounds_20261117/vertical_reorient_full_trial/task.yaml'


def native_program():
    if digest(SCENE.parent.parent/'vertical_reorient_solver/try_plug_pick_sim.py') != BASE_HASH or digest(PRIVATE) != HOOK_HASH:
        raise ValueError('Relative-pose native hook or its original source changed')
    spec = importlib.util.spec_from_file_location('relative_native_pick', PRIVATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def body_pose(sim, name):
    b = sim.data.body(name)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = b.xmat.reshape(3, 3), b.xpos
    return pose


def outlet_frame(sim):
    frame = np.eye(4)
    frame[:3, 3] = [*sim.task.outlets[sim.task.config['target_index']], sim.task.socket_face_z]
    return frame


def pin_points(sim):
    pins = np.vstack([part['points'] for part in sim.task.fixture.parts['yellow_plug']
                      if part['name'].startswith('pin_')])
    return np.column_stack((pins, np.ones(len(pins))))


def grasp_drift(sim, reference, points):
    current = np.linalg.solve(body_pose(sim, 'right/gripper_center'), body_pose(sim, 'task/yellow_plug'))
    return float(np.linalg.norm((points @ (current-reference).T)[:, :3], axis=1).max())


def pin_error(sim, desired):
    actual = np.linalg.solve(body_pose(sim, 'task/strip') @ outlet_frame(sim),
                             body_pose(sim, 'task/yellow_plug'))
    residual = (pin_points(sim) @ (actual-desired).T)[:, :3]
    return dict(lateral_m=float(np.linalg.norm(residual[:, :2], axis=1).max()),
                axial_m=float(abs(residual[:, 2]).max()))


def forbidden_contact(contacts, precontact):
    for contact in contacts:
        pair = {contact['body1'], contact['body2']}
        robot = any(name.startswith('right/') for name in pair)
        if ('task/black_charger' in pair and (robot or 'task/yellow_plug' in pair)
                or precontact and 'task/strip' in pair and (robot or 'task/yellow_plug' in pair)):
            return contact
    return None


class RelativeTask:
    def __init__(self, trial, options):
        self.trial = trial
        self.held_episode = options.get('held_episode') or ('stopped_rgb' if options.get('measurement') else None)
        if options['stage'] != 'precontact':
            raise ValueError('Only a scoped precontact simulation is qualified for this adapter')
        self.precontact_only = True
        self.measured = options.get('measurement')
        self.started = time.monotonic()
        self.events = []
        self.desired = self.grasp = self.points = None
        self.bias = np.eye(4)
        self.corrections = 0
        self.at_goal = False
        self.rejection = None

    def restore_held(self, sim):
        # Diagnostic initial condition only. No writes to object state during a motion.
        if self.measured is not None:
            obs = self.measured
            socket = np.asarray(obs['T_WS'])
            predicted = body_pose(sim, 'task/strip') @ outlet_frame(sim)
            sim.set_observation_state(np.asarray(obs['right_state_si']))
            tool = body_pose(sim, 'right/gripper_center')
            plug = body_pose(sim, 'task/yellow_plug')
            self.events.append(dict(event='INITIALIZE_STOPPED_TWIN', prior_T_WS=predicted.tolist(),
                socket_prediction_residual=np.linalg.solve(predicted,socket).tolist(),
                prior_T_GP=np.linalg.solve(tool,plug).tolist(), observation=obs))
            tip, angle = task_error(np.asarray(obs['T_WG']) @ obs['T_GP'], tool @ obs['T_GP'],
                                     np.asarray(obs['pin_tips_P_m']), np.asarray(obs['insertion_axis_P']))
            tolerance = obs['quality']['task_tolerance']
            if tip > tolerance['max_tip_error_m'] or angle > tolerance['max_axis_error_deg']:
                raise ValueError('Measured tool and native FK disagree at the pin tips')
            strip = socket @ np.linalg.inv(outlet_frame(sim))
            strip_body = sim.model.body('task/strip')
            sim.model.body_pos[strip_body.id] = strip[:3,3]
            mujoco.mju_mat2Quat(sim.model.body_quat[strip_body.id], strip[:3,:3].ravel())
            np.savez(self.trial/'initial_strip.npz', pos=strip[:3,3], quat=sim.model.body_quat[strip_body.id])
            plug = tool @ obs['T_GP']
            address = sim.model.joint('task/yellow_plug/free').qposadr[0]
            sim.data.qpos[address:address+3] = plug[:3,3]
            mujoco.mju_mat2Quat(sim.data.qpos[address+3:address+7],plug[:3,:3].ravel())
            mujoco.mj_forward(sim.model,sim.data)
            save(self.trial/'initial_integration.json',integration_snapshot(sim.model,sim.data))
            return sim.control_targets()
        if self.held_episode not in (HELD, REGRASP):
            raise ValueError('Unreviewed simulated held source')
        source = DATA/'complete_sim/succeed'/self.held_episode/'proprio_stats/privileged.jsonl'
        with source.open() as stream:
            last = None
            for line in stream:
                row = json.loads(line)
                if row['phase'] == 'align':
                    last = row
                elif last is not None:
                    break
        state = last['physics_state']
        if state['mujoco_version'] != mujoco.__version__:
            raise ValueError('Held control requires the archived MuJoCo version')
        mujoco.mj_setState(sim.model, sim.data, np.asarray(state['values']), state['spec'])
        sim.data.time = 0.
        mujoco.mj_forward(sim.model, sim.data)
        save(self.trial/'initial_integration.json', integration_snapshot(sim.model, sim.data))
        self.events.append(dict(event='RESTORE_HELD_SIM_CONTROL', source_episode=self.held_episode,
                                frame=last['frame'], source_sha256=digest(source), prior_trajectory_sha256=digest(
                                    DATA/'self_evolution'/self.held_episode/'trajectory.jsonl')))
        return sim.control_targets()

    def check_step(self, sim, phase, check):
        if self.grasp is not None and phase in ('transport','align'):
            drift = grasp_drift(sim,self.grasp,self.points)
            if drift > sim.task.config['slot_clearance_m']:
                self.rejection=dict(reason='Plug moved in the gripper',phase=phase,
                                    max_pin_displacement_m=drift,sim_time=sim.sim_time)
                raise ValueError('Measured grasp changed during the precontact segment')
        collision = forbidden_contact(check['contacts'], phase not in ('insert','release','retreat','fold'))
        if collision:
            self.rejection = dict(reason='Forbidden object contact', phase=phase, contact=collision,
                                  sim_time=sim.sim_time)
            raise ValueError('Forbidden object contact: '+collision['body1']+' / '+collision['body2'])
        if phase in ('lift','reorient','transport') and not self.held_episode:
            tilt = float(np.rad2deg(sim.read().extras['task']['metrics']['tilt_rad']))
            if tilt > 1.:
                self.rejection = dict(reason='Existing upright gate', phase=phase, tilt_deg=tilt,
                                      limit_deg=1., sim_time=sim.sim_time)
                raise ValueError('Simulated lift/transport exceeded the existing 1 degree upright bound')

    def observe_target(self, sim, phase, strip, grasp, native_desired, read_body):
        offset = outlet_frame(sim)
        socket = strip @ offset
        self.desired = np.linalg.solve(offset, native_desired)
        before = pin_error(sim, self.desired)
        obs = dict(valid=True, T_WS=socket.tolist(), T_GP=grasp.tolist(), source='sim.privileged_body_pose',
                   timestamp_s=sim.sim_time, calibration_version=digest(self.trial/'task.yaml'),
                   quality={'scope':'native simulation geometry only', 'physical_error_bound_m':None})
        if self.measured is not None:
            # A stopped RGB sample can initialize a single no-contact prediction only.
            if any(e['event']=='OBSERVE_TARGET' for e in self.events):
                raise ValueError('Another synchronized RGB boundary observation is required')
            obs = self.measured
            socket, grasp = np.asarray(obs['T_WS']), np.asarray(obs['T_GP'])
        if self.grasp is None:
            self.grasp, self.points = np.array(grasp), pin_points(sim)
        self.events.append(dict(event='OBSERVE_TARGET', phase=phase, observation=obs,
                                T_SP_desired=self.desired.tolist(), before=before,
                                prior_gripper_pose=read_body('right/gripper_center').tolist()))
        return socket, grasp, self.bias @ self.desired

    def log_goal(self, grip):
        self.events[-1]['T_WG_goal'] = grip.tolist()

    def precontact(self, sim):
        error = pin_error(sim, self.desired)
        limit = sim.task.config['slot_clearance_m']
        passed = max(error.values()) <= limit
        self.events.append(dict(event='MEASURE_PRECONTACT', after=error, limit_m=limit, passed=passed))
        if passed:
            self.at_goal = True
            return True
        if self.corrections >= 2 or self.measured is not None:
            raise ValueError('Precontact residual outside clearance; correction budget or RGB observation exhausted')
        actual = np.linalg.solve(body_pose(sim,'task/strip') @ outlet_frame(sim),
                                 body_pose(sim,'task/yellow_plug'))
        self.bias = self.desired @ np.linalg.inv(actual) @ self.bias
        self.corrections += 1
        return False

    def finish(self, sim, phase, error):
        save(self.trial/'relative_result.json', dict(task_id=self.trial.name, events=self.events,
            native_sha256=HOOK_HASH, native_base_sha256=BASE_HASH, rejection=self.rejection,
            source_aligned=self.at_goal, last_phase=phase, native_error=error,
            wall_s=time.monotonic()-self.started, hardware_commands=0))


def read_stopped_rgb(path):
    try:
        return _read_stopped_rgb(path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return dict(valid=False, reason='Stopped right-arm state incomplete', missing=[str(exc)])


def _read_stopped_rgb(path):
    from relative_pose import measure_rgb
    import cv2
    bundle = json.loads(path.read_text())
    rgb = None
    if bundle.get('frame'):
        pixels = cv2.imread(str(path.parent/bundle['frame']))
        if pixels is not None:
            rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    cal = json.loads((path.parent/bundle['calibration']).read_text()) if bundle.get('calibration') else {}
    obs = measure_rgb(rgb, cal, bundle.get('T_WG'), bundle.get('exposure_s'))
    obs['source_frame_sha256'] = digest(path.parent/bundle['frame']) if rgb is not None else None
    if obs['valid']:
        fk_s, skew = bundle.get('fk_time_s'), cal.get('max_pair_skew_s')
        if (not all(isinstance(x, (float,int)) and np.isfinite(x) for x in (fk_s,skew))
                or skew < 0 or bundle.get('arm') != 'right' or bundle.get('target_index') != 3
                or bundle.get('tool_frame') != 'right/gripper_center' or bundle.get('world_frame') != 'simulation_world'
                or obs['source'] != 'fiducial_assisted' or bundle.get('phase') != 'precontact'
                or bundle.get('camera_id') != cal.get('camera_id') or not cal.get('camera_id')
                or bundle.get('clock') != 'recorded_monotonic'
                or abs(float(bundle['fk_time_s'])-obs['timestamp_s']) > float(cal['max_pair_skew_s'])):
            return dict(valid=False, reason='Stopped RGB identity, phase or FK pairing rejected',
                        missing=['same fixed camera, precontact phase and synchronized exposure/FK'])
        # A single exposure cannot establish that the plug is staying rigid in the fingers.
        previous = bundle.get('previous')
        q = np.asarray(bundle.get('right_state_si'),dtype=float)
        if not previous or q.shape != (7,) or not np.isfinite(q).all():
            return dict(valid=False,reason='Stopped right-arm state incomplete',
                        missing=['two adjacent synchronized RGB exposures and six measured joints (rad) plus width (m)'])
        gap = obs['timestamp_s'] - previous['exposure_s']
        prev_fk = previous['fk_time_s']
        if not (0 < gap <= cal.get('max_stopped_interval_s',0) and abs(prev_fk-previous['exposure_s']) <= skew):
            return dict(valid=False,reason='Exposure or FK pairing conflict',missing=['adjacent synchronized stopped frames'])
        image=cv2.imread(str(path.parent/previous['frame']))
        prev=measure_rgb(cv2.cvtColor(image,cv2.COLOR_BGR2RGB) if image is not None else None,
                         cal,previous.get('T_WG'),previous['exposure_s'])
        if not prev['valid']:
            return prev
        tips, axis = np.asarray(cal['pin_tips_P_m']),np.asarray(cal['insertion_axis_P'])
        # Socket displacement is evaluated at plug-tip points expressed in the socket frame.
        sp = np.linalg.solve(obs['T_WS'],np.asarray(bundle['T_WG']) @ obs['T_GP'])
        tol=obs['quality']['task_tolerance']
        for role, basis in [('T_GP',np.eye(4)),('T_WS',sp)]:
            error, deg = task_error(prev[role] @ basis,obs[role] @ basis,tips,axis)
            if error > tol['max_tip_error_m'] or deg > tol['max_axis_error_deg']:
                return dict(valid=False,reason='Stopped pose measurements conflict',missing=[role+' stable across exposures'])
        obs.update(source_bundle_sha256=digest(path),right_state_si=q.tolist(),T_WG=bundle['T_WG'],
                   pin_tips_P_m=tips.tolist(), insertion_axis_P=axis.tolist(),
                   paired_frames_sha256=[digest(path.parent/previous['frame']),digest(path.parent/bundle['frame'])])

    return obs


def validate_segment(trial, recording, obs, held):
    from trajectory_contract import resample
    from validate_shared_reference import validate
    meta = json.loads((recording/'proprio_stats/metadata.json').read_text())
    rows = [json.loads(s) for s in (trial/'trajectory.jsonl').read_text().splitlines()]
    timeline, q, phases = resample(rows, meta['state_feature_names'])
    ref = trial/'precontact_reference'
    ref.mkdir()
    np.savez(ref/'reference.npz', time=timeline, targets=q, phases=np.asarray(phases))
    initial = trial/'initial_integration.json'
    strip = trial/'initial_strip.npz'
    manifest = dict(initial_strip_sha256=digest(strip) if strip.exists() else None,task_scope='precontact_simulation_only', trajectory_id=digest(ref/'reference.npz'),
        source_trajectory_sha256=digest(trial/'trajectory.jsonl'),
        source_initial_state_sha256=digest(trial/'initial_state.npz'), task_sha256=digest(trial/'task.yaml'),
        initial_integration_sha256=digest(initial) if initial.exists() else None,
        held_control=held,T_SP_desired=obs['T_SP_desired'], T_GP_start=next(
            e['observation']['T_GP'] for e in json.loads((trial/'relative_result.json').read_text())['events']
            if e['event']=='OBSERVE_TARGET'),task_id=trial.name)
    save(ref/'manifest.json', manifest)
    return validate(trial, ref)


def diagnostic_owner():
    import subprocess
    import sys
    sys.path.insert(0,str(ROOT))
    from robot_agent.supervisor import Supervisor
    from learn import retrieve
    cost = {}
    def select_check(packet, tokens, seconds):
        warnings = retrieve('in_hand_pose_mismatch', context=dict(active_arms=['right'], target_index=3,
                            source_profile='original',phase='lift'))
        packet['historical_warnings'] = [dict(id=m['id'],scope=m.get('applicability'),
            limits=m.get('known_limits',[])[:1]) for m in warnings[:2]]
        prompt = ('Choose one stopped-task offline check: compare_geometry or verify_observation_and_frame. '
                  'Return just JSON {"category":"UNKNOWN","checks":["<chosen check>"]}. '
                  'No parameter suggestion or motion. '+json.dumps(packet, ensure_ascii=False))
        request = dict(jsonrpc='2.0', id=1, method='tools/call', params=dict(name='review',
                       arguments=dict(executor_model='gpt-6.2-astra', prompt=prompt)))
        command = [Path.home()/'.local/share/aris-runtime/.venv/bin/python',
                   Path.home()/'.config/aris/deepseek_review.py']
        start = time.monotonic()
        cost.update(transport='installed deepseek-review',usage=None, time_limit_s=45, acceptance_token_limit=16000)
        try:
            completed = subprocess.run(command,input=json.dumps(request)+'\n',text=True,
                capture_output=True,timeout=seconds,check=True)
            response = json.loads(completed.stdout)['result']
            if response.get('isError'):
                raise ValueError('Configured reviewer could not complete the check selection')
            review = json.loads(response['content'][0]['text'])
            usage=review['usage']
            cost.update(model=review['reviewer_model'],usage=usage,result='completed')
            reply = json.loads(review['content'].strip().removeprefix('```json').removesuffix('```').strip())
            if reply.get('checks') not in (['compare_geometry'],['verify_observation_and_frame']):
                raise ValueError('Expected one supported stopped-task check')
            return dict(category=reply['category'],checks=reply['checks'],
                        input_tokens=usage['prompt_tokens'],output_tokens=usage['completion_tokens'])
        except (subprocess.SubprocessError, OSError, KeyError, TypeError, ValueError) as exc:
            cost.setdefault('result',type(exc).__name__)
            raise ValueError('Bounded remote diagnosis unavailable') from exc
        finally:
            cost['wall_s']=time.monotonic()-start
    return Supervisor(select_check,max_calls=1,token_limit=16000,seconds=45),cost


def bounded_check(trial, stopped):
    import sys
    sys.path.insert(0, str(ROOT))
    from robot_agent.contracts import Event, FailureReport, Reference
    report_path = trial/'report.json'
    native = json.loads(report_path.read_text())
    ref = Reference('dataset', str(report_path.relative_to(DATA)), digest(report_path))
    native_version=json.loads((trial/'relative_result.json').read_text())['native_sha256']
    event = Event(trial.name,'original',native_version,digest(trial/'task.yaml'),digest(__file__),
                  'plug_insertion', yaml.safe_load((trial/'task.yaml').read_text())['target_index'],
                  native['last_phase'],str(native['frames']),native['frames']/30,'sim_time','verified',
                  'simulator','status','sim.native_report','observed',stopped,(ref,),
                  kind='RIGHT_PRECONTACT_STOP',status='OBSERVED')
    failure = FailureReport(trial.name,[event],event,'RIGHT_PRECONTACT_STOP',False,(ref,))
    supervisor, cost = diagnostic_owner()
    answer=supervisor.diagnose(failure, {'stopped_reason': stopped})
    checks=answer.get('checks') or ['verify_observation_and_frame']
    # The model can select an evidence read, never a motion or a retry.
    check=checks[0]
    data=json.loads((trial/'relative_result.json').read_text())
    if check == 'compare_geometry':
        from validate_shared_reference import load, MujocoBackend, configure_rigid_collection
        from grasp_contact_probe import contact_snapshot
        sim = MujocoBackend(dict(load(), task_scene=yaml.safe_load((trial/'task.yaml').read_text())))
        try:
            configure_rigid_collection(sim)
            with np.load(trial/'final_state.npz') as final:
                for field in ('qpos','qvel','ctrl'):
                    getattr(sim.data,field)[:] = final[field]
            mujoco.mj_forward(sim.model,sim.data)
            probe = contact_snapshot(sim)
            evidence = dict(tool='grasp_contact_probe.contact_snapshot',state='restored native end state',
                            contact_count=len(probe['contacts']), native_rejection=data['rejection'],
                            final_state_sha256=digest(trial/'final_state.npz'))
        finally:
            sim.close()
    else:
        with (trial/'trajectory.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        evidence=dict(tool='native frame and phase reconciliation', native_error=native['error'],
                      frames_reported=native['frames'],frames_recorded=len(rows),
                      last_recorded_phase=rows[-1]['phase'] if rows else None, native_rejection=data['rejection'])
    return dict(supervisor_decision=answer['decision'], root_cause='UNKNOWN', capsule=answer['capsule'],
                reasoning=answer.get('diagnosis'), check_executed=check, evidence=evidence,
                model_calls=supervisor.calls, model_cost=cost, final_decision='STOP', retry=False)


def stop_measurement(observation, bundle):
    from learn import record
    out, result = record('right_relative_pose', 'Stopped right-only RGB qualification', 'offline_rgb')
    decision = dict(task_id=result['experiment_id'], decision='STOP_MISSING_M0', observation=observation,
                    hardware_commands=0, model_calls=0, simulation_started=False)
    if observation['reason'] != 'missing RGB measurement inputs' and 'incomplete' not in observation['reason']:
        supervisor,cost = diagnostic_owner()
        packet = dict(task_id=decision['task_id'], stage='stopped_rgb', status='CONFLICT',
                      missing=observation['missing'], source_sha256=observation.get('source_frame_sha256'))
        selected = supervisor.investigate(packet,dict(decision='ESCALATE_SLOW'))
        check=(selected.get('checks') or ['verify_observation_and_frame'])[0]
        if check=='compare_geometry':
            verified=read_stopped_rgb(bundle)
            evidence=dict(valid=verified['valid'],reason=verified.get('reason'),missing=verified.get('missing'))
        else:
            source=json.loads(bundle.read_text())
            path=bundle.parent/source['frame']
            evidence=dict(frame_sha256=digest(path) if path.is_file() else None,
                          clock=source.get('clock'),fk_time_s=source.get('fk_time_s'),
                          exposure_s=source.get('exposure_s'))
        decision.update(decision='STOP_MEASUREMENT_CONFLICT',model_calls=supervisor.calls,
                        slow=dict(check_executed=check,model_cost=cost,evidence=evidence))
    save(out/'decision.json',decision)
    checkpoint(out, decision)
    print(json.dumps(decision))
    return decision


def consume_trial(out, result, case, stage):
    from episode_format import supplement, validate
    trial = DATA/'self_evolution'/result['experiment_id']
    path = trial/'relative_result.json'
    t0 = time.monotonic()
    summary = dict(task_id=trial.name, stage='precontact', hardware_commands=0, model_calls=0,
                   full_task_success=None, held_control=case['relative_pose']['held_control'])
    try:
        if not path.exists() or result.get('process_returncode') != 0:
            raise ValueError('Native run or recorder incomplete; inspect local stdout')
        source = json.loads(path.read_text())
        if source['task_id'] != trial.name:
            raise ValueError('Relative task ID mismatch')
        episode = None
        if result['observed_effects'][0]['frames']:
            episode, = (stage/'complete_sim/failed').iterdir()
            supplement(episode,trial)
            summary['recording'] = validate(episode,trial,expected_success=False)
            summary['recording_path'] = str(episode.relative_to(DATA))
        summary['raw_observation_sha256'] = digest(path)
        summary['native_phase'] = source['last_phase']
        if source['source_aligned']:
            observed = next(e for e in reversed(source['events']) if e['event']=='OBSERVE_TARGET')
            replay = validate_segment(trial,episode,observed,summary['held_control'])
            first = next(e for e in source['events'] if e['event']=='OBSERVE_TARGET')
            summary.update(corrections=sum(e['event']=='MEASURE_PRECONTACT' for e in source['events']),
                           before=first['before'], source_after=source['events'][-1]['after'],
                           replay_after=replay.get('precontact_error'), reference_sha256=replay['reference_sha256'],
                           replay_wall_s=replay['wall_seconds'], all_substeps_valid=replay['all_substeps_valid'])
            if replay.get('stage_success'):
                summary['decision']='PRECONTACT_VALIDATED'
                summary['next']='STOP_AT_PRECONTACT'
            else:
                summary['reason']=replay.get('first_anomaly') or replay.get('error') or 'Frozen precontact replay rejected'
        else:
            summary['reason']=source['rejection'] or source['native_error'] or 'Precontact not measured'
        if 'reason' in summary:
            summary['slow']=bounded_check(trial,summary['reason'])
            summary.update(decision='STOP_REJECTED',model_calls=summary['slow']['model_calls'])
    except (ValueError, OSError, AssertionError, KeyError) as exc:
        summary.update(decision='STOP_INCOMPLETE',reason=str(exc))
    summary['wall_s'] = result['execution_time'] + time.monotonic() - t0
    if trial.is_dir():
        save(trial/'task_decision.json',summary)
    # Keep the precontact prefix out of full-task success/failure statistics.
    result.update(task_success=None,validation_status=summary['decision'],result='diagnostic',
                  next_improvement=summary.get('next') or summary.get('reason'),stage=summary)
    from learn import finish
    finish(out,result)
    save(out/'task_decision.json',summary)
    checkpoint(out, summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def checkpoint(out, decision):
    from datetime import datetime, timezone
    import fcntl
    # Several independent simulation workers can finish at the same time.
    lock = ROOT/'.local/relative_pose_checkpoint.lock'
    lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        path = ROOT/'evolution/current_state.json'
        state = json.loads(path.read_text())
        entry = state['relative_pose']
        decisions=entry.setdefault('decisions', {})
        if decision['task_id'] not in decisions:
            entry['sim_trials']=entry.get('sim_trials',0)+int('observation' not in decision)
            entry['runtime_model_attempts']=entry.get('runtime_model_attempts',0)+decision['model_calls']
        decisions[decision['task_id']] = dict(
            decision=decision['decision'], model_calls=decision['model_calls'],
            evidence=str((out/('decision.json' if 'observation' in decision else 'task_decision.json')).relative_to(ROOT)))
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        tmp = path.with_suffix('.tmp')
        save(tmp,state)
        tmp.replace(path)

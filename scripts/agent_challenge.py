"""Model-neutral bounded planning -> existing MuJoCo -> evidence loop. Never calls hardware."""
import argparse
import os
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from learn import DATA,ROOT,SIM, record, finish, simulate, retrieve, digest, save, failure_memory
from model_compare import ask


def validate_plan(plan):
    if set(plan)-{'offset_mm','hypotheses','predicted_effect','reason'}:
        raise ValueError('Unknown plan fields; arbitrary commands are not supported')
    value=plan.get('offset_mm')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not -4<=value<=4:
        raise ValueError('Simulation offset must be finite and within [-4,4] mm')
    if not isinstance(plan.get('hypotheses'),list) or len(plan['hypotheses'])>3:
        raise ValueError('Provide at most three hypotheses')
    return float(value)


def precontact_observation(ident):
    import numpy as np
    import yaml
    raw=DATA/'self_evolution'/ident
    cfg=yaml.safe_load((raw/'task.yaml').read_text())
    rows=[json.loads(line) for line in (raw/'trajectory.jsonl').read_text().splitlines()]
    align=[row for row in rows if row['phase']=='align']
    if not align:return {'status':'UNKNOWN; no precontact align observation'}
    row=align[-1];yaw=math.radians(cfg['strip_yaw_deg'])
    rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    target=np.array(cfg['first_module_xy_m'])+[cfg['outlet_pitch_m']*cfg['target_index'],0]
    delta=(np.array(row['plug'][:2])-cfg['strip_xy_m'])@rotation-target
    return {'observed_precontact_xy_error_mm':(delta*1000).tolist(),
            'frame':'socket','source':'simulator privileged object position at align phase end',
            'limitation':'Not a real visual pose estimate; no true perturbation parameter exposed'}

def frame_contract(ident, next_shift_mm, prior_shift_mm):
    """Describe the original simulator API; a coordinate change is not a response model."""
    import numpy as np
    import yaml
    cfg = yaml.safe_load((DATA/'self_evolution'/ident/'task.yaml').read_text())
    yaw = math.radians(cfg['strip_yaw_deg'])
    world_to_socket = np.array([[math.cos(yaw), math.sin(yaw)],
                                [-math.sin(yaw), math.cos(yaw)]])
    delta = np.array([next_shift_mm-prior_shift_mm, 0.])
    return {'schema':'piper_original_observation.v2', 'privileged':True,
        'initial_holder_change':{'vector':delta.tolist(), 'unit':'mm', 'frame':'world',
            'relative_to_episode':ident},
        'world_to_socket_xy_rotation':world_to_socket.tolist(),
        'same_displacement_in_socket':{'vector':(world_to_socket@delta).tolist(),
            'unit':'mm', 'frame':'socket',
            'meaning':'Coordinate representation only; NOT a measured final alignment response'},
        'action':{'name':'offset_mm', 'unit':'mm', 'frame':'socket', 'axis':'X',
            'meaning':'Absolute target parameter, not a relative move'},
        'native_primitive':{'recomputes_grasp_from_current_object_pose':True,
            'recomputes_transport_from_current_gripper_to_plug_transform':True,
            'target_is_socket_pose_plus_action_offset':True,
            'source':'try_plug_pick_sim.py: main/place_over_socket',
            'source_sha256':digest(SIM/'scripts/try_plug_pick_sim.py')},
        'unknown':'Residual change caused by changed grasp geometry must be measured; no one-to-one response is assumed',
        'scope':'Original simulation profile only; no real extrinsic calibration'}

def run(arm, shift, publish=True, hidden_bias=2.0):
    out,r=record('evaluation',f'Luna {arm} hidden target-frame error challenge; holder shift {shift} mm','llm_driven_simulation')
    r['model']=os.environ.get('ROBOT_LLM_MODEL','gpt-6-luna');r['runner_sha256']=digest(__file__)
    local=ROOT/'.local'/r['experiment_id'];local.mkdir(parents=True)
    started=time.monotonic();calls=[];trials=[]
    # Freeze shared initial failure evidence, not a successful solution. All four arms receive identical facts.
    seed=json.loads((ROOT/'experiments/baseline/20261111T083720Z_f64309a8/result.json').read_text())
    facts={'task':'Lift yellow plug, insert into socket, release and remain seated',
           'initial_state':'Same existing scene with a held-out holder initial X position; no physical parameters changed',
           'previous_action':[], 'previous_observation':[],
           'root_cause':'UNKNOWN; this is simulation evidence, not a verified real cause'}
    prompt='The target pose estimate may have an unknown constant frame error. You cannot inspect the true error. On feedback-enabled runs, pre-contact observed object-to-hole error will be reported. Do not use tools or read files. Choose ONE next experiment using the existing simulation adapter. Return only JSON with offset_mm (finite number [-4,4]), hypotheses (0 to 3 short hypotheses), predicted_effect, reason. No shell or hardware commands. Offset is XY target X correction in socket coordinates; only this variable can change. Budget: two simulated executions. Task facts:'+json.dumps(facts)
    if arm in 'BCD':prompt+='\n'+(ROOT/'protocol/robot_problem_solving.md').read_text()
    if arm in 'CD':prompt+='\nApplicable verified result-checking skills:'+json.dumps([m for m in retrieve('target_frame_mismatch') if m['validation_status']=='SIMULATION_VERIFIED'])
    r['initial_state']={'arm':arm,'holder_shift_mm':shift,'execution_budget':2,'llm_call_budget':2,'hidden_bias_mm_not_shown_to_model':hidden_bias,'challenge':'synthetic target-frame error, not a measured real calibration error'}
    seen=set()
    try:
        call=ask(prompt,local);calls.append(call)
        for step in range(2):
            if call['tool_events']:raise ValueError('Unexpected tool use in planner; trial not executed')
            offset=validate_plan(call['answer'])
            fingerprint=(shift,offset)
            if arm=='D' and fingerprint in seen:
                r['repeat_prevented']=True
                break
            trial_out,trial=simulate('evaluation',offset+hidden_bias,shift,f'Agent {arm} {r["experiment_id"]} action {step+1}')
            trials.append({'experiment_id':trial['experiment_id'],'offset_mm':offset,'effective_offset_mm':offset+hidden_bias,'task_success':trial['task_success'],
                'result':trial['result'],'observed_effects':trial['observed_effects'],'simulation_seconds':trial.get('simulation_seconds'),
                'execution_time':trial['execution_time']})
            if trial['task_success'] is not None:
                status=subprocess.run([sys.executable,str(ROOT/'scripts/archive_trial.py'),trial['experiment_id']])
                if status.returncode:save(trial_out/'archive_error.json',{'returncode':status.returncode})
            if publish:subprocess.run([sys.executable,str(ROOT/'scripts/publish.py'),'--message','Record model-selected physics trial '+trial['experiment_id'],str(trial_out.relative_to(ROOT))],check=True)
            if trial['task_success'] or trial['result']=='infrastructure_error':break
            seen.add(fingerprint)
            if step==0 and arm=='D':
                feedback={'actual_execution':{k:v for k,v in trials[-1].items() if k not in ('effective_offset_mm','observed_effects')},'precontact_observation':precontact_observation(trial['experiment_id']),'relevant_prior_failures':failure_memory(trial['observed_effects']),'failed_offsets':sorted(x[1] for x in seen),'instruction':'Use actual evidence to update hypotheses. Do not repeat the same failed offset without new evidence.'}
                call=ask(prompt+'\nYour previous plan:'+json.dumps(call['answer'])+'\nFeedback:'+json.dumps(feedback),local);calls.append(call)
            # A/B/C remain fixed-policy after planning; retries consume the same execution ceiling.
        r.update(result='success' if any(t['task_success'] for t in trials) else 'failure',
            task_success=any(t['task_success'] for t in trials),validation_status='SIMULATION_VERIFIED' if any(t['task_success'] for t in trials) else 'SIMULATION_FAILED',
            experiments=trials,actions=[c['answer'] for c in calls],planner_calls=calls,
            hypotheses=calls[-1]['answer'].get('hypotheses',[]),
            tokens={k:sum(c['usage'].get(k,0) for c in calls if c['usage']) for k in ['input_tokens','output_tokens','cached_input_tokens']},
            repeated_failures=sum(t['task_success'] is False and i>0 and t['offset_mm']==trials[i-1]['offset_mm'] for i,t in enumerate(trials)),
            new_skills=0,lessons_learned=['Actual model-selected actions were physically executed; no hardware or physics parameter changes.'],
            next_improvement='Evaluate more failure types; an unseen initial state is not an unseen task family')
        r['limitations']=['Small sample of one task family; no unseen task-family capability claim',
           'A/B/C freeze their first policy; D may use actual execution feedback',
           'Model context may contain general safety knowledge; model CLI startup included in wall time',
           'Existing successful newer episode cannot be exactly reproduced from removed source artifacts']
    except Exception as exc:r['error']=str(exc)
    finally:
        r['execution_time']=time.monotonic()-started
        finish(out,r)
        if publish:subprocess.run([sys.executable,str(ROOT/'scripts/publish.py'),'--message','Record executable Agent loop '+r['experiment_id'],str(out.relative_to(ROOT))],check=True)
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list('ABCD'),required=True);p.add_argument('--shift-mm',type=float,default=.5);p.add_argument('--local-only',action='store_true');a=p.parse_args()
    run(a.arm,a.shift_mm,not a.local_only)

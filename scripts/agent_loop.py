"""Model-neutral bounded planning -> existing MuJoCo -> evidence loop. Never calls hardware."""
import argparse
import os
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from learn import ROOT, record, finish, simulate, retrieve, digest, save, failure_memory
from model_compare import ask
from alignment_skill import context_for, plan_with_memory


def validate_plan(plan):
    if set(plan)-{'offset_mm','hypotheses','predicted_effect','reason'}:
        raise ValueError('Unknown plan fields; arbitrary commands are not supported')
    value=plan.get('offset_mm')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not -4<=value<=4:
        raise ValueError('Simulation offset must be finite and within [-4,4] mm')
    if not isinstance(plan.get('hypotheses'),list) or len(plan['hypotheses'])>3:
        raise ValueError('Provide at most three hypotheses')
    return float(value)


def compatible_seed_history(profile, target):
    return profile=='original' and str(target)=='3'

def run(arm, shift, publish=True, planner_policy=None):
    out,r=record('evaluation',f'Luna {arm} executable repair loop; target {os.environ.get("PIPER_TARGET_INDEX","3")}; holder shift {shift} mm','llm_driven_simulation')
    r['model']=os.environ.get('ROBOT_LLM_MODEL','gpt-6-luna');r['runner_sha256']=digest(__file__)
    local=ROOT/'.local'/r['experiment_id'];local.mkdir(parents=True)
    started=time.monotonic();calls=[];trials=[]
    policy=planner_policy or json.loads((ROOT/'evolution/active.json').read_text()).get('planner_policy','llm')
    if arm!='D':policy='llm'
    r['actual_planner_policy']=policy
    # Use historical evidence only when task and source profile match.
    seed=json.loads((ROOT/'experiments/baseline/20261111T083720Z_f64309a8/result.json').read_text())
    seed_compatible=compatible_seed_history(os.environ.get('PIPER_SIM_PROFILE','original'),os.environ.get('PIPER_TARGET_INDEX','3'))
    if policy=='compiled_alignment_v1' and seed_compatible:
        seed=json.loads((ROOT/'experiments/research_stage3/20261112T075231Z_2a125acc/result.json').read_text())
    facts={'execution_profile':os.environ.get('PIPER_SIM_PROFILE','original'),
           'prior_evidence_scope':'Only same-profile outlet-3 evidence is included; other source/target histories are excluded',
           'task':'Lift yellow plug, insert into socket, release and remain seated',
           'target_index':int(os.environ.get('PIPER_TARGET_INDEX','3')),
           'mode':('UNKNOWN_TASK_MODE: no target-specific verified motion skill; decompose goal and test the available existing insertion primitive' if os.environ.get('PIPER_TARGET_INDEX','3')!='3' else 'KNOWN_TASK'),
           'initial_state':'Same existing scene with a held-out holder initial X position; no physical parameters changed',
           'previous_action':seed['actions'] if seed_compatible else [],'previous_observation':seed['observed_effects'] if seed_compatible else [],
           'root_cause':'UNKNOWN; this is simulation evidence, not a verified real cause'}
    prompt='Do not use tools or read files. Choose ONE next experiment using the existing simulation adapter. Return only JSON with offset_mm (finite number [-4,4]), hypotheses (0 to 3 short hypotheses), predicted_effect, reason. No shell or hardware commands. Offset is XY target X correction in socket coordinates; only this variable can change. Budget: two simulated executions. Task facts:'+json.dumps(facts)
    if arm in 'BCD':prompt+='\n'+(ROOT/'protocol/robot_problem_solving.md').read_text()
    if arm in 'CD':prompt+='\nApplicable verified result-checking skills:'+json.dumps([m for m in retrieve('plug_not_seated') if m['validation_status']=='SIMULATION_VERIFIED'])
    context=context_for(seed,shift) if policy=='compiled_alignment_v1' and seed_compatible else {}
    if context:prompt+='\nTyped applicability evidence:'+json.dumps(context)
    r['initial_state']={'target_index':int(os.environ.get('PIPER_TARGET_INDEX','3')),'mode':facts['mode'],'arm':arm,'holder_shift_mm':shift,'execution_budget':2,'llm_call_budget':2,'seed_failure':seed['experiment_id'] if seed_compatible else None,'incompatible_history_excluded':not seed_compatible}
    seen=set()
    try:
        call=plan_with_memory(prompt,local,policy,context);calls.append(call)
        for step in range(2):
            if call['tool_events']:raise ValueError('Unexpected tool use in planner; trial not executed')
            offset=validate_plan(call['answer'])
            fingerprint=(shift,offset)
            if arm=='D' and fingerprint in seen:
                r['repeat_prevented']=True
                break
            from benchmark_collection import run_case
            trial=run_case(dict(name=f'{r["experiment_id"]} action {step+1}',offset_mm=offset,shift_mm=shift),
                inline=os.environ.get('PIPER_SIM_PROFILE','original')=='original',label=f'Agent {arm}',kind='evaluation',publish=publish)
            trials.append({'experiment_id':trial['experiment_id'],'offset_mm':offset,'task_success':trial['task_success'],
                'result':trial['result'],'observed_effects':trial['observed_effects'],'simulation_seconds':trial.get('simulation_seconds'),
                'execution_time':trial['execution_time'],'collection_component_seconds':trial['collection_component_seconds'],
                'recording_verified':trial['recording_verified'],'causal_diagnosis':trial.get('causal_diagnosis')})
            if not trial['recording_verified']:raise RuntimeError('Physical trial preserved but complete recording not verified')
            if trial['task_success'] or trial['result']=='infrastructure_error':break
            seen.add(fingerprint)
            if step==0 and arm=='D':
                feedback={'actual_execution':trials[-1],'relevant_prior_failures':failure_memory(trial['observed_effects']),'failed_offsets':sorted(x[1] for x in seen),'instruction':'Use actual evidence to update hypotheses. Do not repeat the same failed offset without new evidence.'}
                context=context_for(trial,shift) if policy=='compiled_alignment_v1' and seed_compatible else {}
                call=plan_with_memory(prompt+'\nYour previous plan:'+json.dumps(call['answer'])+'\nFeedback:'+json.dumps(feedback)+'\nTyped applicability evidence:'+json.dumps(context),local,policy,context);calls.append(call)
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
    except Exception as exc:
        r['error']=str(exc);r['experiments']=trials;r['planner_calls']=calls
    finally:
        r['execution_time']=time.monotonic()-started
        finish(out,r)
        if publish:subprocess.run([sys.executable,str(ROOT/'scripts/publish.py'),'--message','Record executable Agent loop '+r['experiment_id'],str(out.relative_to(ROOT))],check=True)
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list('ABCD'),required=True);p.add_argument('--shift-mm',type=float,default=.5);p.add_argument('--local-only',action='store_true');p.add_argument('--planner-policy',choices=['llm','compiled_alignment_v1']);a=p.parse_args()
    run(a.arm,a.shift_mm,not a.local_only,a.planner_policy)

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


def compatible_seed_history(profile, target, robot_arms='dual'):
    return robot_arms=='dual' and profile=='original' and str(target)=='3'


def rejected_tilts(seed,memories):
    scope=dict(task_config_sha256=seed.get('initial_state',{}).get('task_config_sha256'),
        source_profile=seed['actions'][0].get('profile'),active_arms=seed.get('active_arms'),
        native_sha256=seed.get('program_version',{}).get('source_sha256',{}).get('scripts/try_plug_pick_sim.py'))
    return [m['exact_rejection']['value'] for m in memories if (m.get('exact_rejection') or {}).get('parameter')=='grasp_tilt_deg'
            and all(value is not None and m['exact_rejection'].get(key)==value for key,value in scope.items())]


def repair_diagnosis(seed):
    """Expose existing trace/IK tools to the planner, without promoting a repair."""
    from diagnose_retention import diagnose_episode
    native=seed['observed_effects'][0]
    retention=seed.get('causal_diagnosis') or diagnose_episode(seed)
    context=dict(source_episode=seed['experiment_id'],retention=retention,
        baseline=dict(offset_mm=seed['actions'][0]['offset_mm'][0],
                      grasp_tilt_deg=native.get('grasp_tilt_deg',0.)),
        allowed_grasp_tilt_deg=[],root_cause='UNKNOWN',privileged_information_used=True,
        prior_failure_warnings=[{k:m.get(k) for k in ('id','content','applicability','source_profile','source_episodes','validation_status','exact_rejection')}
                               for m in failure_memory(seed['observed_effects'])],
        memory_scope='Failure warnings only. Different profiles/embodiments are analogies, not current-scene facts or replayable motions.')
    context['rejected_grasp_tilt_deg']=rejected_tilts(seed,context['prior_failure_warnings'])
    if retention['status']=='GRASP_RETENTION_LOSS_OBSERVED':
        context['next_unknown']='Grasp loss precedes planning error; do not diagnose a new transport IK repair from the detached-object state.'
    elif native.get('last_phase')=='transport' and 'Unreachable pose' in str(native.get('error')):
        from probe_transport_ik import probe
        baseline=context['baseline']['grasp_tilt_deg']
        deltas=[0.]+[x-baseline for x in (-10.,-5.,0.,5.,10.) if x!=baseline and abs(x-baseline)<=10]
        result=probe(seed['experiment_id'],[native['preinsert_gap_m']],deltas)
        context.update(probe_experiment_id=result['experiment_id'],probe_result=result['result'],
            diagnostic_seconds=result['execution_time'],counterfactual_limit=result.get('counterfactual_limit'))
        if result['result']=='success':
            context['ik_screen']=[{k:x[k] for k in ('hypothetical_grasp_tilt_delta_deg','ik_success','position_error_m','orientation_error_deg')}
                                  for x in result['experiments']]
            context['allowed_grasp_tilt_deg']=[baseline+x['hypothetical_grasp_tilt_delta_deg'] for x in result['experiments']
                if x['ik_success'] and x['hypothetical_grasp_tilt_delta_deg']!=0
                and baseline+x['hypothetical_grasp_tilt_delta_deg'] not in context['rejected_grasp_tilt_deg']]
        else:context['probe_error']=result.get('error','UNKNOWN')
    return context


def validate_diagnostic_plan(plan,context):
    keys={'offset_mm','grasp_tilt_deg','stop','hypotheses','predicted_effect','reason'}
    if set(plan)-keys or sum(k in plan for k in ('offset_mm','grasp_tilt_deg','stop'))!=1:
        raise ValueError('Choose one existing parameter or stop; no arbitrary commands')
    if not isinstance(plan.get('hypotheses'),list) or len(plan['hypotheses'])>3:
        raise ValueError('Provide at most three hypotheses')
    if 'stop' in plan:
        if plan['stop'] is not True:raise ValueError('Stop must be true')
        return None
    if 'offset_mm' in plan:
        value=validate_plan(plan);parameter='offset_mm'
    else:
        value=plan['grasp_tilt_deg'];parameter='grasp_tilt_deg'
        if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
            or not -10<=value<=10 or value not in context['allowed_grasp_tilt_deg']):
            raise ValueError('Grasp tilt must have current-source native IK screening evidence')
    case=dict(context['baseline']);case[parameter]=value;case['repair_parameter']=parameter
    if parameter=='offset_mm' and case['grasp_tilt_deg']==0:case.pop('grasp_tilt_deg')
    return case


def run(arm, shift, publish=True, planner_policy=None, seed_episode=None, max_executions=2):
    if max_executions not in (1,2):raise ValueError('Execution budget must be one or two')
    robot_arms=os.environ.get('PIPER_ACTIVE_ARMS','dual')
    if robot_arms not in ('dual','right'):raise ValueError('Physical simulation arms must be dual or right')
    out,r=record('evaluation',f'Luna {arm} executable repair loop; target {os.environ.get("PIPER_TARGET_INDEX","3")}; holder shift {shift} mm','llm_driven_simulation')
    r['model']=os.environ.get('ROBOT_LLM_MODEL','gpt-6-luna');r['runner_sha256']=digest(__file__)
    local=ROOT/'.local'/r['experiment_id'];local.mkdir(parents=True)
    started=time.monotonic();calls=[];trials=[];diagnoses=[]
    policy=planner_policy or json.loads((ROOT/'evolution/active.json').read_text()).get('planner_policy','llm')
    if arm!='D' or (robot_arms=='right' and policy=='compiled_alignment_v1'):policy='llm'
    r['actual_planner_policy']=policy
    # Use historical evidence only when task and source profile match.
    seed=json.loads((ROOT/'experiments/baseline/20261111T083720Z_f64309a8/result.json').read_text())
    seed_compatible=compatible_seed_history(os.environ.get('PIPER_SIM_PROFILE','original'),os.environ.get('PIPER_TARGET_INDEX','3'),robot_arms)
    if seed_episode:
        matches=list((ROOT/'experiments').glob('*/'+seed_episode+'/result.json'))
        if len(matches)!=1:raise ValueError('Expected one seed Episode result')
        candidate=json.loads(matches[0].read_text())
        if candidate.get('active_arms',['left','right'])!=(['right'] if robot_arms=='right' else ['left','right']):raise ValueError('Seed embodiment mismatch')
        if candidate['initial_state'].get('target_index')!=int(os.environ.get('PIPER_TARGET_INDEX','3')):raise ValueError('Seed target mismatch')
        from episode_format import source_for
        source_for(candidate)
        seed=candidate;seed_compatible=True
    if policy=='compiled_alignment_v1' and seed_compatible:
        seed=json.loads((ROOT/'experiments/research_stage3/20261112T075231Z_2a125acc/result.json').read_text())
    facts={'active_arms':['right'] if robot_arms=='right' else ['left','right'],
           'execution_profile':os.environ.get('PIPER_SIM_PROFILE','original'),
           'prior_evidence_scope':'Seed evidence must match source, target and physical arms; no right-arm motion skill is yet promoted',
           'task':'Lift yellow plug, insert into socket, release and remain seated',
           'target_index':int(os.environ.get('PIPER_TARGET_INDEX','3')),
           'mode':('UNKNOWN_TASK_MODE: no target-specific verified motion skill; decompose goal and test the available existing insertion primitive' if os.environ.get('PIPER_TARGET_INDEX','3')!='3' else 'KNOWN_TASK'),
           'initial_state':'Existing scene with requested holder initial X position; no physical parameters changed',
           'previous_action':seed['actions'] if seed_compatible else [],'previous_observation':seed['observed_effects'] if seed_compatible else [],
           'root_cause':'UNKNOWN; this is simulation evidence, not a verified real cause'}
    action_spec=('Return only JSON with exactly ONE of offset_mm (finite [-4,4]), grasp_tilt_deg (one screened absolute angle), or stop:true; plus hypotheses (0 to3), predicted_effect and reason. Change one existing parameter. Screened IK is only a hypothesis: actual grasp/transport/insertion must be tested. Stop if no supported low-cost next experiment exists.'
                 if policy=='diagnose_then_repair' else
                 'Return only JSON with offset_mm (finite number [-4,4]), hypotheses (0 to 3 short hypotheses), predicted_effect, reason. Offset is XY target X correction in socket coordinates; only this variable can change.')
    prompt='Do not use tools or read files. Choose ONE next experiment using the existing simulation adapter. No shell or hardware commands. '+action_spec+f' Budget: {max_executions} simulated executions. Task facts:'+json.dumps(facts)
    if arm in 'BCD' and policy!='bounded_tools_v1':prompt+='\n'+(ROOT/'protocol/robot_problem_solving.md').read_text()
    if arm in 'CD' and robot_arms=='dual' and policy!='bounded_tools_v1':prompt+='\nApplicable verified result-checking skills:'+json.dumps([m for m in retrieve('plug_not_seated') if m['validation_status']=='SIMULATION_VERIFIED'])
    if seed_episode and policy!='bounded_tools_v1':
        from agent_challenge import precontact_observation,frame_contract
        prompt+='\nPrecontact observation:'+json.dumps(precontact_observation(seed['experiment_id']))
        prompt+='\nCoordinate contract:'+json.dumps(frame_contract(seed['experiment_id'],shift,seed['initial_state'].get('holder_x_shift_mm',0)))
    context=context_for(seed,shift) if policy=='compiled_alignment_v1' and seed_compatible else {}
    if context:prompt+='\nTyped applicability evidence:'+json.dumps(context)
    r['initial_state']={'target_index':int(os.environ.get('PIPER_TARGET_INDEX','3')),'mode':facts['mode'],'arm':arm,'robot_arms':robot_arms,'holder_shift_mm':shift,'execution_budget':max_executions,'llm_call_budget':max_executions,'seed_failure':seed['experiment_id'] if seed_compatible else None,'incompatible_history_excluded':not seed_compatible}
    seen=set()
    try:
        if policy=='diagnose_then_repair':
            if not seed_compatible:raise ValueError('Diagnostic policy requires a compatible recorded failure')
            context=repair_diagnosis(seed);diagnoses.append(context)
            prompt+='\nMeasured diagnostic evidence and scoped prior failures:'+json.dumps(context)
        if policy=='bounded_tools_v1':
            if not seed_compatible:raise ValueError('Tool policy requires a compatible recorded Episode')
            from bounded_tools import explore
            from benchmark_collection import run_case
            def execute_case(case):
                return run_case(dict(case,name=f'{r["experiment_id"]} bounded action',shift_mm=shift),
                    inline=os.environ.get('PIPER_SIM_PROFILE','original')=='original',
                    label=f'Agent {arm}',kind='evaluation',publish=publish)
            exploration=explore(seed,local,execute_case,max_calls=4*max_executions,
                max_tools=3*max_executions,max_executions=max_executions,shift_mm=shift)
            calls.extend(exploration['planners'])
            diagnoses.extend(c['applicability'] for c in calls)
            for trial,event in zip(exploration['experiments'],exploration['feedback']):
                trials.append(dict(event['observed']['parameters'],**{k:trial.get(k) for k in
                    ('experiment_id','task_success','result','observed_effects','simulation_seconds',
                     'execution_time','collection_component_seconds','recording_verified','causal_diagnosis')}))
            r['exploration']=exploration
            r['pending_real_comparisons']=exploration['pending_real_comparisons']
            r['initial_state']['llm_call_budget']=4*max_executions
            r['initial_state']['diagnostic_tool_budget']=3*max_executions
        else:
            call=plan_with_memory(prompt,local,policy,context,include_catalog=robot_arms!='right')
            calls.append(call)
        for step in range(0 if policy=='bounded_tools_v1' else max_executions):
            if call['tool_events']:raise ValueError('Unexpected tool use in planner; trial not executed')
            if policy=='diagnose_then_repair':
                case=validate_diagnostic_plan(call['answer'],context)
                if case is None:r['stopped_unknown']=True;break
            else:case={'offset_mm':validate_plan(call['answer'])}
            offset=case['offset_mm'];fingerprint=(shift,offset,case.get('grasp_tilt_deg',0.),case.get('grasp_height_m'),case.get('align_gap_m'))
            if arm=='D' and fingerprint in seen:
                r['repeat_prevented']=True
                break
            from benchmark_collection import run_case
            trial=run_case(dict(case,name=f'{r["experiment_id"]} action {step+1}',shift_mm=shift),
                inline=os.environ.get('PIPER_SIM_PROFILE','original')=='original',label=f'Agent {arm}',kind='evaluation',publish=publish)
            trials.append({'experiment_id':trial['experiment_id'],'offset_mm':offset,'grasp_tilt_deg':case.get('grasp_tilt_deg',0.),'task_success':trial['task_success'],
                'grasp_height_m':case.get('grasp_height_m'),'align_gap_m':case.get('align_gap_m'),
                'result':trial['result'],'observed_effects':trial['observed_effects'],'simulation_seconds':trial.get('simulation_seconds'),
                'execution_time':trial['execution_time'],'collection_component_seconds':trial['collection_component_seconds'],
                'recording_verified':trial['recording_verified'],'causal_diagnosis':trial.get('causal_diagnosis')})
            if not trial['recording_verified']:raise RuntimeError('Physical trial preserved but complete recording not verified')
            if trial['task_success'] or trial['result']=='infrastructure_error':break
            seen.add(fingerprint)
            if step+1<max_executions and arm=='D':
                feedback={'actual_execution':trials[-1],'relevant_prior_failures':failure_memory(trial['observed_effects']) if robot_arms=='dual' else [],'failed_offsets':sorted(x[1] for x in seen),'instruction':'Use actual evidence to update hypotheses. Do not repeat the same failed offset without new evidence.'}
                if seed_episode and policy!='bounded_tools_v1':feedback['precontact_observation']=precontact_observation(trial['experiment_id'])
                context=context_for(trial,shift) if policy=='compiled_alignment_v1' and seed_compatible else {}
                if policy=='diagnose_then_repair':context=repair_diagnosis(trial);diagnoses.append(context)
                call=plan_with_memory(prompt+'\nYour previous plan:'+json.dumps(call['answer'])+'\nFeedback:'+json.dumps(feedback)+'\nTyped applicability evidence:'+json.dumps(context),local,policy,context,include_catalog=robot_arms!='right')
                calls.append(call)
            # A/B/C remain fixed-policy after planning; retries consume the same execution ceiling.
        r.update(result='success' if any(t['task_success'] for t in trials) else 'failure',
            task_success=any(t['task_success'] for t in trials) if trials else None,
            validation_status=('SIMULATION_VERIFIED' if any(t['task_success'] for t in trials) else 'SIMULATION_FAILED') if trials else 'NO_PHYSICAL_EXECUTION',
            experiments=trials,actions=[c['answer'] for c in calls],planner_calls=calls,
            hypotheses=calls[-1]['answer'].get('hypotheses',[]) if calls else [],
            tokens=({k:sum(c['usage'].get(k,0) for c in calls) for k in ['input_tokens','output_tokens','cached_input_tokens']}
                    if all(c.get('usage') is not None for c in calls) else None),
            repeated_failures=sum(t['task_success'] is False and i>0 and
                all(t.get(k)==trials[i-1].get(k) for k in ('offset_mm','grasp_tilt_deg','grasp_height_m','align_gap_m')) for i,t in enumerate(trials)),
            new_skills=0,lessons_learned=['Actual model-selected actions were physically executed; no hardware or physics parameter changes.' if trials else
                'No physical experiment selected; UNKNOWN is not a measured physical failure or success.'],
            next_improvement='Evaluate more failure types; an unseen initial state is not an unseen task family')
        r['limitations']=['Small sample of one task family; no unseen task-family capability claim',
           'A/B/C freeze their first policy; D may use actual execution feedback',
           'Model context may contain general safety knowledge; model CLI startup included in wall time',
           'Existing successful newer episode cannot be exactly reproduced from removed source artifacts']
        if policy=='bounded_tools_v1' and exploration['status'] in ('EXPLORATION_ERROR','RECORDING_ERROR','EXECUTION_ERROR'):
            r.update(result='infrastructure_error',validation_status=exploration['status'],error=exploration['error'])
        if policy=='bounded_tools_v1':r['task_success']=exploration['task_success']
    except Exception as exc:
        r['error']=str(exc);r['experiments']=trials;r['planner_calls']=calls
    finally:
        r['diagnostic_contexts']=diagnoses
        r['execution_time']=time.monotonic()-started
        finish(out,r)
        if publish:
            paths=[str(out.relative_to(ROOT))]+['experiments/research_stage4/'+d['probe_experiment_id'] for d in diagnoses if 'probe_experiment_id' in d]
            subprocess.run([sys.executable,str(ROOT/'scripts/publish.py'),'--message','Record executable Agent loop '+r['experiment_id'],*paths],check=True)
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--robot-arms',choices=['dual','right'],default='dual');p.add_argument('--seed-episode');p.add_argument('--arm',choices=list('ABCD'),required=True);p.add_argument('--shift-mm',type=float,default=.5);p.add_argument('--local-only',action='store_true');p.add_argument('--planner-policy',choices=['llm','compiled_alignment_v1','diagnose_then_repair','bounded_tools_v1']);p.add_argument('--max-executions',type=int,choices=[1,2],default=2);a=p.parse_args()
    os.environ['PIPER_ACTIVE_ARMS']=a.robot_arms
    run(a.arm,a.shift_mm,not a.local_only,a.planner_policy,a.seed_episode,a.max_executions)

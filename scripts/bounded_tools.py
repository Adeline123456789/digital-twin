"""JSON tool requests -> fixed offline functions -> evidence -> bounded existing repair plan."""
import json
import math
import time
from pathlib import Path
from evidence_cache import EvidenceCache
from episode_evidence import summary, window
from learn import ROOT
from model_compare import ask, ModelCallError

TOOLS = {
    'episode_summary': {},
    'trace_window': {'center':'nonnegative frame', 'radius':'integer 0..15'},
    'inspect_contact': {'frame':'nonnegative frame'},
    'diagnose_repair': {},
}


def baseline_case(seed):
    action=seed['actions'][0];native=seed['observed_effects'][0]
    case=dict(offset_mm=action['offset_mm'][0],grasp_tilt_deg=native.get('grasp_tilt_deg',0.))
    for key in ('grasp_height_m','align_gap_m','start_right_deg'):
        value=seed.get('comparison_case',{}).get(key)
        if value is None:value=action.get(key,native.get(key))
        if value is not None:case[key]=value
    if 'start_right_deg' not in case and 'start_right_joint_degrees' in seed['initial_state']:
        case['start_right_deg']=seed['initial_state']['start_right_joint_degrees']
    return case


def validate_tool_plan(answer,context):
    from agent_loop import validate_diagnostic_plan
    additional={'grasp_height_m':(.010,.025),'align_gap_m':(.020,.025)}
    chosen=set(answer)&set(additional)
    if not chosen:return validate_diagnostic_plan(answer,context)
    if len(chosen)!=1:raise ValueError('Change only one existing parameter')
    key=chosen.pop()
    if set(answer)-{key,'hypotheses','predicted_effect','reason'}:
        raise ValueError('Unknown plan fields or multiple changed parameters')
    value=answer[key];lo,hi=additional[key]
    if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:
        raise ValueError('Parameter outside existing simulation adapter bounds')
    if not isinstance(answer.get('hypotheses'),list) or len(answer['hypotheses'])>3:
        raise ValueError('Provide at most three hypotheses')
    return dict(context['baseline'],**{key:value},repair_parameter=key)


class EpisodeTools:
    def __init__(self, seed, cache):
        self.seed, self.cache = seed, cache
        self.context = dict(baseline=baseline_case(seed), allowed_grasp_tilt_deg=[])

    def call(self, name, args):
        if name not in TOOLS or not isinstance(args,dict) or set(args) != set(TOOLS[name]):
            raise ValueError('Unknown tool or arguments')
        ident = self.seed['experiment_id']
        if name == 'episode_summary': return summary(ident,self.cache)
        if name == 'trace_window': return window(ident,args['center'],args['radius'],self.cache)
        if name == 'inspect_contact':
            if type(args['frame']) is not int or args['frame'] < 0: raise ValueError('Invalid frame')
            from contact_evidence import inspect_frame
            return inspect_frame(ident,args['frame'],self.cache)
        from agent_loop import repair_diagnosis
        self.context = repair_diagnosis(self.seed)
        self.context['baseline']=baseline_case(self.seed)
        return dict(value=self.context,cache=None)


def total_usage(calls):
    keys = ('input_tokens','output_tokens','cached_input_tokens')
    known = {key:sum(c['usage'].get(key,0) for c in calls if c.get('usage')) for key in keys}
    return known if all(c.get('usage') is not None for c in calls) else None


def prompt_events(events):
    """Keep full tool evidence in the run log; omit duplicate geometry and hash manifests from prompts."""
    output=[]
    for event in events:
        if 'result' not in event:
            output.append(event);continue
        value=event['result']['value']
        if event['request']['tool']=='inspect_contact':
            geoms={g['name']:g for c in value['contacts'] for g in c['geoms']}
            value=dict(value,contacts=[{k:v for k,v in c.items() if k!='geoms'} for c in value['contacts']],geoms=geoms)
        cache=event['result'].get('cache')
        output.append(dict(request=event['request'],value=value,
                           evidence_key=cache['key'] if cache else None))
    return output


def plan(seed, local, *, model=None, max_calls=4, max_tools=3, seconds=90,
         guidance='', ask_fn=ask, dispatcher=None, cache_directory=None, initial_summary=False, feedback=(), scene=None):
    if not 1 <= max_calls <= 8 or not 0 <= max_tools <= 6 or not 1 <= seconds <= 240:
        raise ValueError('Planner budget outside supported bounds')
    started=time.monotonic();calls=[];events=[];seen=set();bootstrap=None
    cache=EvidenceCache(cache_directory or ROOT/'.local/evidence_cache') if dispatcher is None else None
    dispatch=dispatcher or EpisodeTools(seed,cache)
    answer=dict(stop=True,hypotheses=[],reason='No supported plan within budget',predicted_effect='No new execution')
    status='BUDGET_EXHAUSTED'
    initial=dict(episode_id=seed['experiment_id'], active_arms=seed.get('active_arms'),
        task=seed['task'], target=seed['initial_state'].get('target_index'),
        validation_status=seed['validation_status'], task_success=seed['task_success'],
        current_parameters=dispatch.context['baseline'],
        requested_execution_scene=scene,
        root_cause='UNKNOWN', source='Archived simulation; privileged data available on request')
    contract=('Return {"tool":"name","args":{...},"reason":"short reason"} or '
        '{"plan":{"offset_mm":number OR "grasp_tilt_deg":screened_angle OR "grasp_height_m":number OR "align_gap_m":number OR "stop":true,'
        '"hypotheses":[],"predicted_effect":"...","reason":"..."}}. '
        'One parameter only. Offset [-4,4]mm in socket X; height [0.010,0.025]m; align gap [0.020,0.025]m. Tilt requires diagnose_repair screening. '
        'The host executes your JSON tool requests; these are not native shell tools. '
        'A plan requests the existing complete simulation collector, never hardware. '
        'Inspect uncertainty before proposing a repair; no forced friction assumption.')
    try:
        if initial_summary:
            t=time.monotonic()
            evidence=dispatch.call('episode_summary',{})
            initial['observed_summary']=evidence['value']
            bootstrap=dict(result=evidence,wall_seconds=time.monotonic()-t,
                source='Deterministic initial observation; not a model tool decision')
        for _ in range(max_calls):
            remaining=seconds-(time.monotonic()-started)
            if remaining<=0: status='TIME_BUDGET_EXHAUSTED';break
            prompt=contract+'\n'+json.dumps(dict(state=initial,tools=TOOLS,guidance=guidance,
                budget=dict(calls_remaining=max_calls-len(calls),tools_remaining=max_tools-len(events),seconds_remaining=round(remaining,2)),
                prior_experiments=feedback,evidence=prompt_events(events)),ensure_ascii=False,separators=(',',':'))
            try:
                call=ask_fn(prompt,local,include_catalog=False,profile='bounded_tools_v1',model=model,timeout=remaining)
            except ModelCallError as exc:
                calls.append(exc.call);status='MODEL_ERROR';break
            calls.append(call)
            if call['tool_events']:
                status='NATIVE_TOOL_REJECTED';break
            request=call['answer']
            if not isinstance(request,dict): status='INVALID_REQUEST';break
            if set(request)=={'plan'}:
                if not isinstance(request['plan'],dict): status='INVALID_REQUEST';break
                validate_tool_plan(request['plan'],dispatch.context)
                answer=request['plan'];status='STOPPED_UNKNOWN' if answer.get('stop') else 'PLAN_READY';break
            if set(request) != {'tool','args','reason'} or len(events)>=max_tools:
                status='TOOL_BUDGET_OR_SCHEMA_REJECTED';break
            fingerprint=json.dumps([request['tool'],request['args']],sort_keys=True)
            if fingerprint in seen:
                status='DUPLICATE_TOOL_REJECTED';break
            seen.add(fingerprint)
            t=time.monotonic()
            result=dispatch.call(request['tool'],request['args'])
            events.append(dict(request=request,result=result,wall_seconds=time.monotonic()-t))
    except Exception as exc:
        status='EVIDENCE_OR_REQUEST_ERROR'
        events.append(dict(error_type=type(exc).__name__,error=str(exc)))
    finally:
        if cache is not None:cache.close()
    return dict(answer=answer,usage=total_usage(calls),tool_events=0,planner_kind='bounded_tools_v1',
        wall_seconds=time.monotonic()-started,status=status,calls=calls,tool_trace=events,
        applicability=dispatch.context,model_calls_attempted=len(calls),
        execution_scene=scene,
        initial_observation=bootstrap,
        timing=dict(llm_cli_network_inference_seconds=sum(c['wall_seconds'] for c in calls),
                    tools_seconds=sum(e.get('wall_seconds',0) for e in events),
                    initial_observation_seconds=bootstrap['wall_seconds'] if bootstrap else 0),
        inference_only_seconds=None, research_orchestrator_tokens=None)


def experiment_feedback(result, case):
    """Keep the action/effect link, without raw frames or the full source manifest."""
    return dict(experiment_id=result['experiment_id'],parameters=case,
        task_success=result['task_success'],result=result['result'],
        recording_verified=result['recording_verified'],
        native_outcome=[{k:n.get(k) for k in ('last_phase','error','complete_pick_insert',
            'grasp_height_m','align_gap_m','grasp_tilt_deg')} for n in result['observed_effects']],
        causal_diagnosis=result.get('causal_diagnosis'),
        execution_time=result['execution_time'],collection_component_seconds=result['collection_component_seconds'])


def explore(seed, local, execute_case, *, model=None, guidance='', ask_fn=ask,
            max_calls=4, max_tools=3, max_executions=2, seconds=600, cache_directory=None, shift_mm=None):
    """Shared recipient/runtime loop; the trusted collector owns physics and outcomes.

    Each failed execution becomes the next observation. Call/tool/time budgets span
    the whole loop, not each retry. Teacher revision occurs outside this function.
    """
    if not 1<=max_calls<=8 or not 0<=max_tools<=6 or max_executions not in (1,2) or seconds<=0:
        raise ValueError('Unsupported exploration budget')
    started=time.monotonic();planners=[];experiments=[];feedback=[];seen=set()
    calls_left=max_calls;tools_left=max_tools;status='EXECUTION_BUDGET_EXHAUSTED';error=None
    try:
        for _ in range(max_executions):
            remaining=seconds-(time.monotonic()-started)
            if remaining<1:status='TIME_BUDGET_EXHAUSTED';break
            if calls_left<=0:status='MODEL_BUDGET_EXHAUSTED';break
            scene=None
            if shift_mm is not None:
                from agent_challenge import execution_scene
                scene=execution_scene(seed['experiment_id'],shift_mm)
            call=plan(seed,local,model=model,guidance=guidance,ask_fn=ask_fn,
                max_calls=calls_left,max_tools=tools_left,seconds=min(90,remaining),
                cache_directory=cache_directory,initial_summary=True,feedback=feedback,scene=scene)
            planners.append(call)
            calls_left-=call['model_calls_attempted']
            tools_left-=sum('request' in e for e in call['tool_trace'])
            if call['status']!='PLAN_READY':status=call['status'];break
            case=validate_tool_plan(call['answer'],call['applicability'])
            fingerprint=json.dumps({k:v for k,v in case.items() if k!='repair_parameter'},sort_keys=True)
            if fingerprint in seen:status='REPEATED_EXPERIMENT_REJECTED';break
            if time.monotonic()-started>=seconds:status='TIME_BUDGET_EXHAUSTED';break
            seen.add(fingerprint)
            result=execute_case(case)
            experiments.append(result)
            feedback.append(dict(proposal=call['answer'],observed=experiment_feedback(result,case)))
            if not result['recording_verified']:status='RECORDING_ERROR';break
            if result['result']=='infrastructure_error':status='EXECUTION_ERROR';break
            if result['task_success']:status='SIMULATION_SUCCESS_PENDING_REAL';break
            seed=result
    except Exception as exc:
        status='EXPLORATION_ERROR';error=dict(type=type(exc).__name__,message=str(exc))
    return dict(status=status,error=error,planners=planners,experiments=experiments,feedback=feedback,
        usage=total_usage(planners),wall_seconds=time.monotonic()-started,
        task_success=(any(r['task_success'] is True for r in experiments)
                      if any(r['task_success'] is not None for r in experiments) else None),
        budget_used=dict(model_calls=max_calls-calls_left,tools=max_tools-tools_left,executions=len(experiments)),
        pending_real_comparisons=[r['experiment_id'] for r in experiments if r['task_success'] and r['recording_verified']])

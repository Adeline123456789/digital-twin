"""Explicitly launched, budgeted recipient -> teacher -> recipient simulation comparison.

No hardware path. A design is frozen before model calls; approval is external to this script.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import time
from bounded_tools import explore, baseline_case, prompt_events
from model_compare import ask, ModelCallError
from learn import ROOT, DATA, digest, save, record, finish


class BatchBudget:
    def __init__(self, limits, path, prior=None, started_at=None):
        self.limits=limits;self.path=path;self.started=time.monotonic()
        self.started_at=started_at or datetime.now(timezone.utc).isoformat()
        self.started-=time.time()-datetime.fromisoformat(self.started_at).timestamp()
        self.calls=list(prior['calls']) if prior else []
        self.simulations=list(prior['simulations']) if prior else []
        self.usage_unknown=prior['usage_unknown'] if prior else False

    def write(self):
        save(self.path,dict(limits=self.limits,started_at=self.started_at,calls=self.calls,simulations=self.simulations,
            wall_seconds=time.monotonic()-self.started,reported_tokens=self.tokens(),usage_unknown=self.usage_unknown))

    def tokens(self):
        return sum((c.get('usage') or {}).get('input_tokens',0)+(c.get('usage') or {}).get('output_tokens',0) for c in self.calls)

    def check(self,model=True):
        if time.monotonic()-self.started>=self.limits['seconds']:raise RuntimeError('Batch wall budget exhausted')
        if model and self.usage_unknown:raise RuntimeError('Missing model usage; stop instead of assuming zero cost')
        if model and self.tokens()>=self.limits['tokens']:raise RuntimeError('Observed token threshold reached')

    def model(self,*args,**kwargs):
        self.check()
        if len(self.calls)>=self.limits['model_calls']:raise RuntimeError('Model call budget exhausted')
        kwargs['timeout']=min(kwargs.get('timeout',120),self.limits['seconds']-(time.monotonic()-self.started))
        try:
            call=ask(*args,**kwargs)
        except ModelCallError as exc:
            self.calls.append(exc.call);self.usage_unknown=exc.call['usage'] is None;self.write();raise
        self.calls.append(call);self.usage_unknown=call['usage'] is None;self.write()
        return call

    def simulate(self,case,label):
        from benchmark_collection import run_case
        self.check(model=False)
        if len(self.simulations)>=self.limits['simulations']:raise RuntimeError('Simulation budget exhausted')
        index=len(self.simulations)
        self.simulations.append(dict(label=label,status='STARTED'));self.write()
        result=run_case(case,inline=True,label=label,kind='recipient_feedback',publish=False)
        self.simulations[index]=dict(label=label,status='FINISHED',experiment_id=result['experiment_id'],
            task_success=result['task_success'],recording_verified=result['recording_verified'],
            result=result['result'],execution_time=result['execution_time'],
            collection_component_seconds=result['collection_component_seconds'],recording_report=result.get('recording_report'))
        self.write()
        if not result['recording_verified']:raise RuntimeError('Physical result preserved; recording failed independent verification')
        return result


def teacher_feedback(run):
    """No source tree, held-out cases, raw video or dependency manifests in teacher context."""
    exploration=run.get('exploration')
    return dict(plan=run['planner']['answer'],planner_status=run['planner']['status'],
        tools=prompt_events(run['planner']['tool_trace']),
        execution_feedback=exploration['feedback'] if exploration else [],
        recipient_steps=[dict(plan=c['answer'],status=c['status'],tools=prompt_events(c['tool_trace']),
                             execution_scene=c.get('execution_scene'),
                             initial_observation=(c.get('initial_observation') or {}).get('result',{}).get('value'))
                         for c in exploration['planners']] if exploration else [],
        physical_result=run.get('physical_result'),native_outcome=run.get('native_outcome'),
        scope='Student decisions and actual feedback from development only; not a verified causal diagnosis')


def execute_student(seed,guidance,label,layout,budget,local,student):
    exploration=explore(seed,local,lambda case:budget.simulate(dict(case,name=label,shift_mm=layout),label),
        model=student,max_calls=3,max_tools=2,max_executions=2,seconds=600,
        guidance=guidance,ask_fn=budget.model,cache_directory=local/label/'cache',shift_mm=layout)
    call=exploration['planners'][-1] if exploration['planners'] else dict(
        answer={'stop':True},status=exploration['status'],tool_trace=[])
    run=dict(label=label,source_episode=seed['experiment_id'],planner=call,
        exploration=exploration,physical_result=None)
    if exploration['feedback']:
        result=exploration['feedback'][-1]['observed']
        run.update(physical_result={k:result[k] for k in ('experiment_id','task_success','recording_verified','result')},
            native_outcome=result['native_outcome'],
            collection_component_seconds=sum(e['observed']['collection_component_seconds'] for e in exploration['feedback']))
    return run


def comparison_status(runs):
    completed={'STOPPED_UNKNOWN','SIMULATION_SUCCESS_PENDING_REAL','EXECUTION_BUDGET_EXHAUSTED',
               'MODEL_BUDGET_EXHAUSTED','REPEATED_EXPERIMENT_REJECTED'}
    incomplete=[run['label'] for run in runs if
        (run['exploration']['status'] not in completed if 'exploration' in run else
         run['planner']['status'] not in ('PLAN_READY','STOPPED_UNKNOWN'))]
    return dict(result='infrastructure_error' if incomplete else 'success',task_success=None,
        validation_status='COMPARISON_INCOMPLETE' if incomplete else 'COMPARISON_EXECUTED_NOT_PROMOTED',
        incomplete_arms=incomplete)


def run(design_path,resume=None):
    design=json.loads(design_path.read_text())
    if design['hardware_executions']!=0:raise ValueError('This runner never supports hardware')
    from episode_evidence import locate
    from episode_format import source_for
    report,_,raw=locate(design['source_episode'])
    source=json.loads(report.read_text());source_for(source)
    if source.get('active_arms')!=['right']:raise ValueError('Study is scoped to existing right-arm simulation')
    task=raw/'task.yaml'
    if digest(task)!=design['task_sha256']:raise ValueError('Frozen source task changed')
    os.environ.update(PIPER_ACTIVE_ARMS='right',PIPER_SIM_PROFILE='original',PIPER_TARGET_INDEX=str(source['initial_state']['target_index']),PIPER_TASK_CONFIG=str(task))
    out,r=record('recipient_feedback','Actual recipient feedback and teacher-off layout comparison','llm_driven_simulation')
    local=ROOT/'.local'/r['experiment_id'];local.mkdir(parents=True)
    save(out/'design.json',design)
    prior=json.loads((resume/'ledger.json').read_text()) if resume else None
    previous=json.loads((resume/'result.json').read_text()) if resume else None
    if previous and previous.get('error')!='Invalid teacher guidance':
        raise ValueError('This continuation reuses the rejected teacher-format result only')
    budget=BatchBudget(design['limits'],out/'ledger.json',prior,
                      (prior.get('started_at') or previous['timestamp']) if prior else None)
    inherited_tokens=budget.tokens()
    started=time.monotonic();runs=[]
    guidance=(ROOT/'protocol/robot_problem_solving.md').read_text()
    initial_hash=digest(ROOT/'protocol/robot_problem_solving.md')
    files=['scripts/bounded_tools.py','scripts/contact_evidence.py','scripts/episode_evidence.py',
           'scripts/benchmark_recipient_feedback.py','scripts/model_compare.py','scripts/collect_inline.py',
           'protocol/bounded_tool_instructions.md','scripts/agent_loop.py','scripts/probe_transport_ik.py',
           'scripts/diagnose_retention.py','scripts/episode_format.py','scripts/benchmark_collection.py',
           'scripts/learn.py','scripts/evidence_cache.py','scripts/agent_challenge.py']
    frozen={name:digest(ROOT/name) for name in files}
    frozen_memory={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'memory').rglob('*.json')}
    r.update(initial_state=dict(design_sha256=digest(design_path),tool_sha256=frozen,protocol_sha256=initial_hash,
        source_episode=design['source_episode'],teacher_visible_split='development only',
        recipient_limits=dict(model_calls=3,diagnostic_tools=2,executions=2,seconds=600),
        execution_loop='bounded_tools.explore shared with agent_loop bounded_tools_v1; aggregate per-recipient budget',
        heldout_scope='same task family, new layout; not cross-task-family generalization',
        cross_case_memory_updates=False,memory_snapshot_sha256=frozen_memory),model=design['student'],privileged_information_used=True,
        promotion_status='NOT_PROMOTED',research_orchestrator_tokens=None,human_interventions=0,
        resumed_from=resume.name if resume else None,inherited_tokens=inherited_tokens,
        token_accounting='Cumulative batch ledger; do not sum the resumed parent and child token totals')
    try:
        base=baseline_case(source)
        dev=design['development']
        if resume:
            seed_id=prior['simulations'][0]['experiment_id']
            seed=json.loads(locate(seed_id)[0].read_text())
            before=json.loads((resume/'runs.json').read_text())[0]
        else:
            seed=budget.simulate(dict(base,**dev['parameters'],name='development_seed',shift_mm=dev['layout_shift_mm']),'development seed')
            before=execute_student(seed,guidance,'recipient before teacher',dev['layout_shift_mm'],budget,local,design['student'])
        runs.append(before);save(out/'runs.json',runs)
        prompt=('Revise general tool-use guidance from this recipient execution. Do not solve held-out cases or invent measurements. '
            'Return JSON with guidance (max6000 characters), change_reason, and direct_solution (boolean; true if supplying instance-specific repair values). '
            'Keep evidence checks, uncertainty, applicability and one-variable experiments. No safety, speed or evaluator changes. '
            'Existing guidance:'+guidance+'\nRecipient feedback:'+json.dumps(teacher_feedback(before),separators=(',',':')))
        teacher=prior['calls'][-1] if resume else budget.model(prompt,local,include_catalog=False,profile='bounded_tools_v1',model=design['teacher'])
        proposal=teacher['answer']
        if teacher['tool_events'] or not isinstance(proposal,dict) or set(proposal)!={'guidance','change_reason','direct_solution'}:
            raise ValueError('Teacher must return a bounded guidance proposal, not call native tools')
        if not isinstance(proposal['guidance'],str) or not proposal['guidance'].strip() or type(proposal['direct_solution']) is not bool:
            raise ValueError('Invalid teacher guidance')
        # Length is a prompt target, not a physical validity gate; actual tokens stay in the batch ledger.
        teacher['guidance_exceeded_length_target']=len(proposal['guidance'])>6000
        save(out/'teacher_revision.json',teacher)
        revised=proposal['guidance']
        runs.append(execute_student(seed,revised,'recipient after feedback',dev['layout_shift_mm'],budget,local,design['student']))
        save(out/'runs.json',runs)
        # No teacher call occurs below this line. Tools and guidance are frozen before held-out construction.
        r['heldout_guidance_sha256']=__import__('hashlib').sha256(revised.encode()).hexdigest()
        held=design['heldout']
        held_seed=budget.simulate(dict(base,**held['parameters'],name='heldout_seed',shift_mm=held['layout_shift_mm']),'heldout seed')
        for name,text in [('parent',guidance),('candidate',revised)]:
            if any(digest(ROOT/p)!=h for p,h in frozen.items()):raise RuntimeError('Frozen tool changed during comparison')
            if {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'memory').rglob('*.json')}!=frozen_memory:
                raise RuntimeError('Frozen memory changed during comparison')
            runs.append(execute_student(held_seed,text,'heldout '+name,held['layout_shift_mm'],budget,local,design['student']))
            save(out/'runs.json',runs)
        r.update(comparison_status(runs))
        r['lessons_learned']=['Evaluate paired physical outcomes and full teacher/development costs before retaining guidance.']
    except Exception as exc:r.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        budget.write()
        r.update(experiments=runs,planner_calls=budget.calls,tokens=None if budget.usage_unknown else budget.tokens(),
            new_tokens=None if budget.usage_unknown else budget.tokens()-inherited_tokens,
            execution_time=time.monotonic()-started,simulation_experiments=budget.simulations,
            next_improvement='Audit recipient choices, physical success, recording validity and teacher-inclusive cost; no automatic promotion',
            limitations=['One development and one held-out layout; same task family',
                'Requested endpoint identities; backend weights not independently verified',
                'Token/time thresholds are checked between calls/trials; in-flight work can exceed a threshold',
                'No real movement, weight training or claimed real calibration'])
        finish(out,r)
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--design',required=True,type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
    if not a.execute:raise SystemExit('Review/freeze the design and obtain the new experiment budget before --execute')
    run(a.design,a.resume)

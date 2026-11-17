"""Explicitly launched, budgeted recipient -> teacher -> recipient simulation comparison.

No hardware path. A design is frozen before model calls; approval is external to this script.
"""
import argparse
import json
import os
from pathlib import Path
import time
from bounded_tools import plan, validate_tool_plan, baseline_case, prompt_events
from model_compare import ask, ModelCallError
from learn import ROOT, DATA, digest, save, record, finish


class BatchBudget:
    def __init__(self, limits, path):
        self.limits=limits;self.path=path;self.started=time.monotonic()
        self.calls=[];self.simulations=[];self.usage_unknown=False

    def write(self):
        save(self.path,dict(limits=self.limits,calls=self.calls,simulations=self.simulations,
            wall_seconds=time.monotonic()-self.started,reported_tokens=self.tokens(),usage_unknown=self.usage_unknown))

    def tokens(self):
        return sum((c.get('usage') or {}).get('input_tokens',0)+(c.get('usage') or {}).get('output_tokens',0) for c in self.calls)

    def check(self):
        if time.monotonic()-self.started>=self.limits['seconds']:raise RuntimeError('Batch wall budget exhausted')
        if self.usage_unknown:raise RuntimeError('Missing model usage; stop instead of assuming zero cost')
        if self.tokens()>=self.limits['tokens']:raise RuntimeError('Observed token threshold reached')

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
        self.check()
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
    return dict(plan=run['planner']['answer'],planner_status=run['planner']['status'],
        tools=prompt_events(run['planner']['tool_trace']),
        physical_result=run.get('physical_result'),native_outcome=run.get('native_outcome'),
        scope='Student decisions and actual feedback from development only; not a verified causal diagnosis')


def execute_student(seed,guidance,label,layout,budget,local,student):
    call=plan(seed,local,model=student,max_calls=3,max_tools=2,seconds=90,
              guidance=guidance,ask_fn=budget.model,cache_directory=local/label/'cache')
    run=dict(label=label,source_episode=seed['experiment_id'],planner=call,physical_result=None)
    case=validate_tool_plan(call['answer'],call['applicability'])
    if case is None:return run
    result=budget.simulate(dict(case,name=label,shift_mm=layout),label)
    run.update(physical_result=dict(experiment_id=result['experiment_id'],task_success=result['task_success'],
        recording_verified=result['recording_verified'],result=result['result']),
        native_outcome=[{k:native.get(k) for k in ('last_phase','error','complete_pick_insert','grasp_height_m','align_gap_m','grasp_tilt_deg')}
                        for native in result['observed_effects']],collection_component_seconds=result['collection_component_seconds'])
    return run


def run(design_path):
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
    budget=BatchBudget(design['limits'],out/'ledger.json')
    started=time.monotonic();runs=[]
    guidance=(ROOT/'protocol/robot_problem_solving.md').read_text()
    initial_hash=digest(ROOT/'protocol/robot_problem_solving.md')
    files=['scripts/bounded_tools.py','scripts/contact_evidence.py','scripts/episode_evidence.py',
           'scripts/benchmark_recipient_feedback.py','scripts/model_compare.py','scripts/collect_inline.py']
    frozen={name:digest(ROOT/name) for name in files}
    r.update(initial_state=dict(design_sha256=digest(design_path),tool_sha256=frozen,protocol_sha256=initial_hash,
        source_episode=design['source_episode'],teacher_visible_split='development only',
        heldout_scope='same task family, new layout; not cross-task-family generalization',
        cross_case_memory_updates=False),model=design['student'],privileged_information_used=True,
        promotion_status='NOT_PROMOTED',research_orchestrator_tokens=None,human_interventions=0)
    try:
        base=baseline_case(source)
        dev=design['development']
        seed=budget.simulate(dict(base,**dev['parameters'],name='development_seed',shift_mm=dev['layout_shift_mm']),'development seed')
        before=execute_student(seed,guidance,'recipient before teacher',dev['layout_shift_mm'],budget,local,design['student'])
        runs.append(before);save(out/'runs.json',runs)
        prompt=('Revise general tool-use guidance from this recipient execution. Do not solve held-out cases or invent measurements. '
            'Return JSON with guidance (max6000 characters), change_reason, and direct_solution (boolean; true if supplying instance-specific repair values). '
            'Keep evidence checks, uncertainty, applicability and one-variable experiments. No safety, speed or evaluator changes. '
            'Existing guidance:'+guidance+'\nRecipient feedback:'+json.dumps(teacher_feedback(before),separators=(',',':')))
        teacher=budget.model(prompt,local,include_catalog=False,profile='bounded_tools_v1',model=design['teacher'])
        proposal=teacher['answer']
        if teacher['tool_events'] or not isinstance(proposal,dict) or set(proposal)!={'guidance','change_reason','direct_solution'}:
            raise ValueError('Teacher must return a bounded guidance proposal, not call native tools')
        if not isinstance(proposal['guidance'],str) or not 0<len(proposal['guidance'])<=6000 or type(proposal['direct_solution']) is not bool:
            raise ValueError('Invalid teacher guidance')
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
            runs.append(execute_student(held_seed,text,'heldout '+name,held['layout_shift_mm'],budget,local,design['student']))
            save(out/'runs.json',runs)
        r.update(result='success',task_success=None,validation_status='COMPARISON_EXECUTED_NOT_PROMOTED',
            lessons_learned=['Evaluate paired physical outcomes and full teacher/development costs before retaining guidance.'])
    except Exception as exc:r.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        budget.write()
        r.update(experiments=runs,planner_calls=budget.calls,tokens=None if budget.usage_unknown else budget.tokens(),
            execution_time=time.monotonic()-started,simulation_experiments=budget.simulations,
            next_improvement='Audit recipient choices, physical success, recording validity and teacher-inclusive cost; no automatic promotion',
            limitations=['One development and one held-out layout; same task family',
                'Requested endpoint identities; backend weights not independently verified',
                'Token/time thresholds are checked between calls/trials; in-flight work can exceed a threshold',
                'No real movement, weight training or claimed real calibration'])
        finish(out,r)
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--design',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise SystemExit('Review/freeze the design and obtain the new experiment budget before --execute')
    run(a.design)

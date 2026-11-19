"""Single owner of replay state. Slow responses can select offline investigations only."""
import time
from .fast_loop import FastLoop
from .memory_router import capsule

CHECKS = ('verify_observation_and_frame','compare_geometry','compare_retention','inspect_insert_path')
CATEGORIES = {'POLICY_ERROR','WORLD_MODEL_ERROR','PERCEPTION_OR_TIME_ERROR','CONTROL_OR_INTERFACE_ERROR','UNKNOWN'}


class Supervisor:
    def __init__(self, slow=None, max_calls=0, token_limit=4096, seconds=15):
        if not 0 <= max_calls <= 2 or seconds<=0 or token_limit<0:
            raise ValueError('Invalid diagnostic budget')
        self.fast=FastLoop()
        self.slow=slow
        self.max_calls,self.tokens,self.deadline=max_calls,token_limit,time.monotonic()+seconds
        self.calls=self.input_tokens=self.output_tokens=0
        self.planner_seconds=0.
        self.usage_complete=True
        self.trace=['IDLE','PLAN','EXECUTE_FAST']
        self.seen=set()

    def diagnose(self,report,context=None):
        e=report.first_divergence
        if not e and report.status=='NO_DIVERGENCE_OBSERVED' and report.task_success:
            self.trace.append('VERIFY')
            return dict(decision='CONTINUE')
        if not e:
            self.trace+=['HALT_OR_HOLD','TRIAGE_FAST']
            return dict(decision='UNKNOWN',capsule=capsule(report))
        self.trace+=['HALT_OR_HOLD','TRIAGE_FAST']
        key=(e.experiment_id,e.kind,e.action_id)
        if key in self.seen:
            return dict(decision='UNKNOWN',diagnosis='UNKNOWN',reason='already reviewed this event')
        self.seen.add(key)
        decision=self.fast.decide(e,context)
        memory=decision.get('retrieval',{})
        packet=capsule(report,memory.get('matches',()),counterexamples=memory.get('veto',()))
        answer=dict(decision=decision['decision'],capsule=packet,plan=memory.get('plan'))
        if answer['decision']=='APPLY_VERIFIED_SIM_SKILL':
            self.trace+=['EXECUTE_FAST','VERIFY']
            return answer
        if answer['decision']!='ESCALATE_SLOW':
            return answer
        if context and 'stopped_reason' in context:
            packet['stopped_reason'] = context['stopped_reason']
        return self.investigate(packet, answer)

    def investigate(self, packet, answer):
        self.trace.append('ESCALATE_SLOW')
        remaining=self.deadline-time.monotonic()
        if self.calls>=self.max_calls or not self.slow or self.tokens<=0 or remaining<=0:
            return dict(answer,diagnosis='UNKNOWN',checks=[CHECKS[0]],reason='unavailable, duplicate or budget exhausted')
        self.calls+=1
        start=time.monotonic()
        try:
            reply=self.slow(packet,self.tokens,remaining)
            n_in,n_out=reply['input_tokens'],reply['output_tokens']
            if type(n_in)!=int or type(n_out)!=int or min(n_in,n_out)<0:
                raise ValueError('Missing measured usage')
            self.input_tokens+=n_in
            self.output_tokens+=n_out
            self.tokens-=n_in+n_out
            checks=reply['checks']
            if (self.tokens<0 or time.monotonic()>self.deadline or reply['category'] not in CATEGORIES
                    or not isinstance(checks,list) or len(checks)>3 or any(x not in CHECKS for x in checks)):
                raise ValueError('Budget or diagnosis schema')
            self.trace.append('OFFLINE_VALIDATE')
            return dict(answer,diagnosis=reply['category'],checks=sorted(set(checks),key=CHECKS.index),root_cause_status='UNKNOWN')
        except (KeyError,TypeError,ValueError,TimeoutError,OSError):
            self.tokens=0
            self.usage_complete=False
            return dict(answer,diagnosis='UNKNOWN',checks=[],reason='incomplete slow response')
        finally:
            self.planner_seconds+=time.monotonic()-start

    def review(self,proposal):
        from .gate import EvidenceGate
        self.trace.append('OFFLINE_VALIDATE')
        decision=EvidenceGate().review(proposal)
        self.trace.append('READY_FOR_APPROVAL' if decision.decision=='ACCEPT' else 'HALT_OR_HOLD')
        return decision

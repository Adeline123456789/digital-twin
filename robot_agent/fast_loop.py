"""Offline frame consumer. The frozen retention predicate operates on at most four rows."""
from collections import deque
from dataclasses import replace
from math import isfinite
from .first_divergence import event
from .memory_router import MemoryRouter
from diagnose_retention import analyze_retention


class FastLoop:
    def __init__(self, router=None, fps=30):
        self.router = router or MemoryRouter()
        self.max_interval = 1.5/fps
        self.last = self.identity = self.lift = self.latched = None
        self.window = deque(maxlen=3)

    def decide(self, e, context=None):
        if e.issue() or e.status=='UNKNOWN':
            return dict(decision='UNKNOWN',event=e)
        if e.severity=='danger':
            return dict(decision='HALT_REQUEST',event=e)
        if e.severity=='info':
            return dict(decision='CONTINUE',event=e)
        retrieval=self.router.retrieve_applicable(e,context)
        return dict(decision='APPLY_VERIFIED_SIM_SKILL' if retrieval['plan'] else 'ESCALATE_SLOW',
                    event=e,retrieval=retrieval)

    def update(self,o):
        identity=(o.experiment_id,o.source_profile,o.native_source_hash,o.world_model_version,
                  o.policy_version,o.task_id,o.target_index,o.clock_source)
        problem=o.issue()
        if self.identity is not None and identity!=self.identity:
            problem='VERSION_UNKNOWN'
        if self.last is not None and not problem and not 0 < o.timestamp_mono-self.last <= self.max_interval:
            problem='TIMESTAMP_MISMATCH'
        if problem:
            self.latched=dict(decision='UNKNOWN',event=event(o,problem))
        if self.latched is not None:
            return self.latched
        self.identity,self.last=identity,o.timestamp_mono
        if o.sensor_provenance!='sim.privileged_contact':
            return dict(decision='CONTINUE')
        value=o.observed_value
        row=dict(phase=o.phase,time=o.timestamp_mono,plug=value['plug'],grip=value['grip'])
        if not all(len(v)==3 and all(isfinite(x) for x in v) for v in (row['plug'],row['grip'])):
            self.latched=dict(decision='UNKNOWN',event=event(o,'SENSOR_UNAVAILABLE'))
            return self.latched
        sample=(o,row,dict(contacts=value['contacts']))
        if o.phase=='lift':
            self.lift=sample
        if o.phase not in ('transport','align'):
            self.window.clear()
            return dict(decision='CONTINUE')
        if not self.lift:
            self.latched=dict(decision='UNKNOWN',event=event(o,'SENSOR_UNAVAILABLE'))
            return self.latched
        self.window.append(sample)
        if len(self.window)==3:
            observations,rows,extras=zip(self.lift,*self.window)
            d=analyze_retention(rows,extras)
            if d['status']=='GRASP_RETENTION_LOSS_OBSERVED':
                start=observations[1]
                measured=replace(start,observed_value=d['first_event_distance_mm']/1000,
                                 predicted_value=d['lift_reference_distance_mm']/1000)
                alarm=event(measured,d['status'],status='OBSERVED',first_observed_frame=int(start.action_id),
                            first_observed_time=start.timestamp_mono,confirmation_time=o.timestamp_mono)
                self.latched=self.decide(alarm)
                return self.latched
        return dict(decision='CONTINUE')

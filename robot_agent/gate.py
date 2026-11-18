"""Supervisor-owned review of pinned local evidence. No actuator or active-file writer."""
from dataclasses import dataclass, asdict
from pathlib import Path
from .legacy import ROOT, digest
from .diagnosis import EventBuilder, json_file
from .timeline import aligned_samples


@dataclass(frozen=True)
class CandidateProposal:
    category: str
    mechanism: str
    prediction: str
    counterexample: str
    scope: dict
    previous_active_sha256: str
    implementation: dict
    development: tuple
    heldout: tuple
    rollback_ref: str
    invariants: dict


@dataclass(frozen=True)
class PromotionDecision:
    decision: str
    reason: str
    rows: list
    rollback_ref: str
    scope: dict

    def record(self):
        return asdict(self)


class EvidenceGate:
    def __init__(self,root=ROOT):
        self.root=root

    def review(self,p):
        def finish(decision,reason):
            return PromotionDecision(decision,reason,rows,p.rollback_ref,p.scope)
        rows=[]
        if p.category not in {'policy','sim_world','observation_adapter','diagnostic_rule','skill'}:
            return finish('DEFER_UNKNOWN','Unknown candidate type')
        if digest(self.root/'evolution/active.json')!=p.previous_active_sha256:
            return finish('DEFER_UNKNOWN','Active baseline changed')
        if not all((p.mechanism,p.prediction,p.counterexample,p.scope,p.rollback_ref,p.implementation,p.invariants)):
            return finish('DEFER_UNKNOWN','Incomplete preregistration')
        for path,sha in (p.implementation | p.invariants).items():
            if not self.matches(path,sha):
                return finish('REJECT_AND_ROLLBACK','Implementation or frozen invariant changed')
        dev={r['path'] for r in p.development}
        held={r['path'] for r in p.heldout}
        if dev & held or len(held)!=len(p.heldout):
            return finish('DEFER_UNKNOWN','Overlapping split')
        for split,records in [('development',p.development),('heldout',p.heldout)]:
            for record in records:
                if not self.matches(record['path'],record['sha256']):
                    return finish('DEFER_UNKNOWN','Evidence missing or revised')
                path=self.root/record['path']
                r=json_file(path)
                if (r['actions'][0]['profile']!=p.scope['source_profile'] or
                    r['initial_state']['target_index']!=p.scope['target_index'] or
                    r['program_version']['source_sha256']['scripts/try_plug_pick_sim.py']!=p.scope['native_source_hash']):
                    return finish('DEFER_UNKNOWN','Incompatible source or target')
                d=EventBuilder().build(path)
                rows.append(dict(split=split,episode=r['experiment_id'],success=d.task_success,
                                 diagnosis=d.status,config=r['initial_state']['task_config_sha256']))
                if not r['recording_verified'] or d.status=='UNKNOWN':
                    return finish('DEFER_UNKNOWN','Insufficient observations')
                if p.category in ('policy','sim_world','skill') and d.task_success is False:
                    return finish('REJECT_AND_ROLLBACK','Native complete task failed')
                if p.category=='observation_adapter' and d.task_success and d.status!='NO_DIVERGENCE_OBSERVED':
                    return finish('REJECT_AND_ROLLBACK','Valid recorded success regressed')
        if not held or not dev:
            return finish('DEFER_UNKNOWN','Incomplete independent comparison')
        if p.category!='observation_adapter' or p.scope.get('purpose')!='detect_time_mismatch_in_archived_simulation':
            return finish('DEFER_UNKNOWN','Additional independent physical checks required')
        # The reviewed experiment changes exactly one clock channel in memory on each saved Episode.
        from .legacy import DATA
        import h5py
        for row in rows:
            ident=row['episode']
            if not row['success']:
                return finish('DEFER_UNKNOWN','This adapter comparison requires success controls')
            source=DATA/'self_evolution'/ident/'trajectory.jsonl'
            stats=DATA/'complete_sim/succeed'/ident/'proprio_stats'
            import json
            recorded=[json.loads(x) for x in source.read_text().splitlines()]
            priv=[json.loads(x) for x in (stats/'privileged.jsonl').read_text().splitlines()]
            with h5py.File(stats/'proprio_stats.h5') as f:
                clock=f['sim_time'][:]
            fps=json_file(stats/'metadata.json')['fps']
            row['original_aligned']=aligned_samples(recorded,priv,clock,fps)
            clock[len(clock)//2]+=.2
            row['injected_mismatch_detected']=not aligned_samples(recorded,priv,clock,fps)
            if not (row['original_aligned'] and row['injected_mismatch_detected']):
                return finish('REJECT_AND_ROLLBACK','Clock test regressed')
        return finish('ACCEPT','Offline timeline check only; native policy and physical labels unchanged')

    def matches(self,path,sha):
        path=(self.root/path).resolve()
        return path.is_relative_to(self.root) and path.is_file() and digest(path)==sha

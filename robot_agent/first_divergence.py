from dataclasses import fields
from .contracts import Event, FailureReport, Observation


def event(observation, kind, **detail):
    fact = {f.name: getattr(observation, f.name) for f in fields(Observation)}
    return Event(**fact, kind=kind, **detail)


def first_divergence(events):
    relevant = [e for e in events if e.severity != 'info']
    if not relevant:
        return None
    identity = {(e.experiment_id,e.source_profile,e.native_source_hash,e.world_model_version,
                 e.policy_version,e.task_id,e.target_index,e.clock_source) for e in relevant}
    if len(identity) != 1 or any(e.issue() or e.status == 'UNKNOWN' for e in relevant):
        return None
    # Terminal exceptions can have only a lower time bound, after all sampled events.
    return min(relevant, key=lambda e: (e.first_observed_time is None, e.first_observed_time if e.first_observed_time is not None else e.timestamp_mono))


def report(ident, events, native_success, refs):
    first = first_divergence(events)
    status = first.kind if first else ('UNKNOWN' if any(e.severity!='info' for e in events) else 'NO_DIVERGENCE_OBSERVED')
    return FailureReport(ident, events, first, status, native_success, refs)

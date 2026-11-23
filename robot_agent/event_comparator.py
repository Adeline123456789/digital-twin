"""Compare evidence, never convert unavailable evidence into a negative event."""
from math import isfinite
from .interaction_events import Comparison as C, Truth as T, EventComparison, EventSpec, InteractionEventTrace


def reliable_observation(spec, observation, now, confidence_min=.9):
    return (observation is not None and observation.event_id == spec.event_id
            and observation.clock_valid and observation.evidence_ref is not None
            and isfinite(observation.timestamp) and abs(observation.timestamp - now) <= 1e-8
            and isfinite(observation.confidence) and confidence_min <= observation.confidence <= 1
            and set(spec.observation_requirements) <= set(observation.available_observations)
            and observation.status in (T.TRUE, T.FALSE))


def compare_event(spec, observation, now, matched=(), clock_valid=True, confidence_min=.9):
    def result(status, reason):
        return EventComparison(spec, observation, status, now, reason)
    if not clock_valid or not isfinite(now):
        return result(C.UNRESOLVED, 'Clock discontinuity')
    reliable = reliable_observation(spec, observation, now, confidence_min)
    if reliable and observation.status is T.TRUE:
        if not spec.expected_presence:
            return result(C.UNEXPECTED, 'Forbidden event observed')
        if not set(spec.prerequisites) <= set(matched) or now < spec.earliest_time or now > spec.deadline:
            return result(C.UNEXPECTED, 'Event outside required order or time window')
        return result(C.MATCHED, 'Expected event confirmed')
    if now < spec.earliest_time:
        return result(C.PENDING, 'Window not open')
    if not spec.expected_presence:
        return result(C.MATCHED if reliable else C.UNRESOLVED,
                      'Expected absence observed' if reliable else 'Absence cannot be established')
    if now <= spec.deadline:
        return result(C.PENDING, 'Deadline not passed')
    return result(C.CONTRADICTED if reliable else C.UNRESOLVED,
                  'Reliable absence after deadline' if reliable else 'Required observation unavailable')


class EventComparator:
    def __init__(self, specs, max_interval=1.5 / 30):
        self.specs = tuple(specs)
        self.max_interval = max_interval
        self.trace = InteractionEventTrace()

    def update(self, now, observations):
        trace = self.trace
        first_sample = trace.last_time is None
        if trace.last_time is not None and not 0 < now - trace.last_time <= self.max_interval:
            trace.clock_valid = False
        trace.last_time = now
        results = []
        by_id = {o.event_id: o for o in observations}
        for spec in self.specs:
            # Occurrences remain confirmed; absence predicates are checked every sample.
            if spec.expected_presence and spec.event_id in trace.matched and trace.clock_valid:
                previous = next(c for c in reversed(trace.comparisons) if c.spec.event_id == spec.event_id)
                results.append(previous)
                continue
            comparison = compare_event(spec, by_id.get(spec.event_id), now, trace.matched, trace.clock_valid)
            observation = by_id.get(spec.event_id)
            if (spec.earliest_time <= now <= spec.deadline and not reliable_observation(spec, observation, now)):
                trace.missing_in_window.add(spec.event_id)
            if first_sample and now - spec.earliest_time > self.max_interval:
                trace.missing_in_window.add(spec.event_id)
            if comparison.status is C.CONTRADICTED and spec.event_id in trace.missing_in_window:
                comparison = EventComparison(spec, observation, C.UNRESOLVED, now, 'Incomplete event-window coverage')
            results.append(comparison)
            if comparison.status is C.MATCHED and spec.expected_presence:
                trace.matched.add(spec.event_id)
        known = {spec.event_id for spec in self.specs}
        for observation in observations:
            if observation.event_id not in known:
                spec = EventSpec(observation.event_id, observation.event_id, 'unknown', False, 0., max(0., now))
                results.append(compare_event(spec, observation, now, clock_valid=trace.clock_valid))
        trace.comparisons.extend(results)
        return results

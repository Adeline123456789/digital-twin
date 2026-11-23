from dataclasses import replace
from event_fixtures import *
from robot_agent.interaction_events import EVENT_SEQUENCE
from robot_agent.event_comparator import EventComparator


def test_all_events_enforce_prerequisite_order():
    assert len(EVENT_SEQUENCE) == 9
    specs = [EventSpec(name, name, 'test', True, 0, 1, EVENT_SEQUENCE[max(0,i-1):i])
             for i, name in enumerate(EVENT_SEQUENCE)]
    comparator = EventComparator(specs)
    for i, name in enumerate(EVENT_SEQUENCE):
        rows = comparator.update(i/30, [ObservedEvent(name, T.TRUE, 1., i/30, REF)])
        assert rows[i].status is C.MATCHED
    wrong = EventComparator(specs).update(0, [ObservedEvent(EVENT_SEQUENCE[1], T.TRUE, 1., 0, REF)])
    assert wrong[1].status is C.UNEXPECTED


def test_clock_discontinuity_and_lost_window_are_unresolved():
    s = replace(spec(), earliest_time=0, deadline=.05)
    c = EventComparator([s])
    c.update(0, [observed(T.FALSE, 0)])
    c.update(1/30, [])
    assert c.update(2/30, [observed(T.FALSE, 2/30)])[0].status is C.UNRESOLVED
    assert c.update(-1, [observed(T.TRUE, -1)])[0].status is C.UNRESOLVED

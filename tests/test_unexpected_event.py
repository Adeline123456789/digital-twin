from dataclasses import replace
from event_fixtures import *
from robot_agent.recovery_router import route, Route
from robot_agent.event_comparator import EventComparator


def test_unknown_unexpected_never_automatically_recovers():
    c = compare_event(spec(False), observed(T.TRUE), 1.5)
    assert c.status is C.UNEXPECTED
    assert route(c, within_bounds=True) is Route.UNKNOWN
    early = compare_event(spec(), observed(T.TRUE, .5), .5)
    assert early.status is C.UNEXPECTED
    late = compare_event(spec(), observed(T.TRUE, 2.1), 2.1)
    assert late.status is C.UNEXPECTED
    unknown = replace(observed(T.TRUE, 0.), event_id='UNREGISTERED_EVENT')
    result = EventComparator([spec()]).update(0., [unknown])[-1]
    assert result.status is C.UNEXPECTED
    assert route(result, within_bounds=True) is Route.UNKNOWN

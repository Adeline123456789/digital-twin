from event_fixtures import *


def test_deadline_is_inclusive_and_absence_requires_passed_deadline():
    assert compare_event(spec(), observed(T.TRUE, 2.), 2.).status is C.MATCHED
    assert compare_event(spec(), observed(T.FALSE, 2.), 2.).status is C.PENDING
    assert compare_event(spec(), None, .5).status is C.PENDING
    assert compare_event(spec(), None, 2.1).status is C.UNRESOLVED

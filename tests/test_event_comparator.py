from dataclasses import replace
from event_fixtures import *


def test_all_five_states():
    assert compare_event(spec(), observed(T.FALSE, .5), .5).status is C.PENDING
    assert compare_event(spec(), observed(T.TRUE), 1.5).status is C.MATCHED
    assert compare_event(spec(), observed(T.FALSE, 2.1), 2.1).status is C.CONTRADICTED
    assert compare_event(spec(), observed(T.UNKNOWN, 2.1), 2.1).status is C.UNRESOLVED
    assert compare_event(spec(False), observed(T.TRUE), 1.5).status is C.UNEXPECTED


def test_missing_low_confidence_or_stale_data_cannot_contradict():
    for observation in [None, observed(T.FALSE, 2.1, .2), observed(T.FALSE, 1.5),
                        replace(observed(T.FALSE, 2.1), available_observations=()),
                        replace(observed(T.FALSE, 2.1), evidence_ref=None)]:
        assert compare_event(spec(), observation, 2.1).status is C.UNRESOLVED


def test_expected_absence():
    assert compare_event(spec(False), observed(T.FALSE), 1.5).status is C.MATCHED
    assert compare_event(spec(False), observed(T.UNKNOWN), 1.5).status is C.UNRESOLVED

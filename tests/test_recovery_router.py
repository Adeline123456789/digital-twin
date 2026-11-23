from event_fixtures import *
from robot_agent.recovery_router import route, Route


def test_only_validated_bounded_evidence_can_request_recovery():
    c = compare_event(spec(), observed(T.FALSE, 2.1), 2.1)
    assert route(c, validated_recoveries=('ALIGNMENT_OK',), within_bounds=True) is Route.RESIDUAL
    assert route(c, validated_recoveries=('ALIGNMENT_OK',)) is Route.UNKNOWN
    assert route(c, static_geometry_verified=True, within_bounds=True) is Route.RETARGET
    assert route(c, repeated_structural_evidence=True) is Route.STRUCTURAL_CANDIDATE
    matched = compare_event(spec(), observed(T.TRUE), 1.5)
    assert route(matched, repeated_structural_evidence=True) is Route.NO_ACTION

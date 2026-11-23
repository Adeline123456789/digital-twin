from event_fixtures import *
from robot_agent.recovery_router import route, Route, safety_decision


def test_unknown_blocks_even_requested_recovery_and_structural_route():
    c = compare_event(spec(), observed(T.UNKNOWN, 2.1), 2.1)
    assert route(c, validated_recoveries=('ALIGNMENT_OK',), within_bounds=True,
                 static_geometry_verified=True, repeated_structural_evidence=True) is Route.NO_ACTION
    assert safety_decision(c, critical=True) == 'HOLD'

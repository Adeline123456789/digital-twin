from event_fixtures import *
from robot_agent.recovery_router import route, Route, observation_mode, safety_decision


def test_reacquire_precontact_but_switch_modality_at_contact():
    c = compare_event(spec(), observed(T.UNKNOWN, 2.1), 2.1)
    assert route(c, safe_precontact=True) is Route.REACQUIRE_OBSERVATION
    assert route(c, safe_precontact=True, contact_likely=True) is Route.NO_ACTION
    assert observation_mode(c, contact_likely=True) == 'PROPRIOCEPTIVE_CONTACT_MODE'
    assert safety_decision(c, critical=True) == 'HOLD'

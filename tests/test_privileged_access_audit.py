import pytest
from robot_agent.nominal_reference import PlannerSpy, PrivilegedAccessAudit


def test_planner_and_privileged_attempts_are_counted_and_denied():
    spy = PlannerSpy()
    with pytest.raises(RuntimeError, match='freeze'):
        spy.forbidden()
    assert spy.calls == 1
    audit = PrivilegedAccessAudit()
    for field in ('goal_pose', 'object_pose', 'contact_pairs', 'sim_success'):
        with pytest.raises(PermissionError):
            audit.reject(field)
    assert audit.policy_reads == 4
    assert audit.accesses == dict(goal=1, object=1, contact=1, other=1)

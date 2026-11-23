from robot_agent.memory_router import IsolatedExperimentMemory
from robot_agent.fast_loop import FastLoop
from robot_agent.supervisor import Supervisor


def test_isolated_supervisor_never_constructs_legacy_memory_or_slow():
    router = IsolatedExperimentMemory()
    owner = Supervisor(fast=FastLoop(router=router))
    answer = router.retrieve_applicable(None)
    assert answer['plan'] is None and answer['matches'] == []
    assert owner.max_calls == owner.calls == 0 and owner.slow is None

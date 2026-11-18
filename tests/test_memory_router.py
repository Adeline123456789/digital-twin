from robot_agent.contracts import FailureReport
from robot_agent.memory_router import capsule,index_memory
from robot_agent.first_divergence import event
from test_observation_contract import OBS


def test_capsule_bounds_and_revocation():
    events=[event(OBS,'TARGET_FRAME_RESIDUAL') for _ in range(6)]
    report=FailureReport('id',events,events[-1],'OBSERVED',False,OBS.evidence_refs)
    small=capsule(report,list(range(8)),list(range(5)))
    assert len(small['events'])==3 and len(small['memory_refs'])==5 and len(small['hypotheses'])==3
    index=index_memory()
    assert {'NE011','VS002'} <= index.keys()
    assert all('EXCLUDED' not in card['validation_status'] for card,_ in index.values())

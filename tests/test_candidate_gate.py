from dataclasses import replace
from robot_agent.gate import CandidateProposal,EvidenceGate
from robot_agent.diagnosis import json_file,ROOT
from robot_agent.legacy import digest


def proposal(name='clock'):
    return CandidateProposal(**json_file(ROOT/f'experiments/fast_slow_v2/P3.3/{name}.json'))


def test_wrong_source_changed_invariant_and_duplicate_split():
    gate=EvidenceGate()
    active=ROOT/'evolution/active.json'
    before=active.read_bytes()
    p=proposal()
    assert gate.review(replace(p,previous_active_sha256='stale')).decision=='DEFER_UNKNOWN'
    assert gate.review(replace(p,heldout=p.development)).decision=='DEFER_UNKNOWN'
    assert gate.review(replace(p,scope=dict(p.scope,source_profile='real'))).decision=='DEFER_UNKNOWN'
    assert gate.review(replace(p,invariants={'scripts/learn.py':'bad'})).decision=='REJECT_AND_ROLLBACK'
    assert active.read_bytes()==before

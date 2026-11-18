from dataclasses import replace
from robot_agent.gate import CandidateProposal,EvidenceGate
from robot_agent.diagnosis import json_file,ROOT
from robot_agent.legacy import digest


def proposal(name='clock'):
    return CandidateProposal(**json_file(ROOT/f'experiments/fast_slow_v2/P4.2/{name}.json'))


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

def test_reviewer_adversarial_cases():
    p=proposal()
    gate=EvidenceGate()
    duplicate=dict(p.development[0],path='./'+p.development[0]['path'])
    assert gate.review(replace(p,heldout=[duplicate])).decision=='DEFER_UNKNOWN'
    assert gate.review(replace(p,rollback_ref='bad')).decision=='DEFER_UNKNOWN'
    impl=dict(p.implementation)
    impl['scripts/learn.py']='incorrect'
    assert gate.review(replace(p,implementation=impl)).decision=='REJECT_AND_ROLLBACK'
    record=dict(p.development[0],assets={})
    assert gate.review(replace(p,development=[record])).decision=='DEFER_UNKNOWN'


def test_physical_gate_rejects_known_safety_and_attachment_contradictions(monkeypatch):
    r=dict(experiment_id='fixture',observed_effects=[dict(max_penetration_m=.001)])
    assert EvidenceGate.physical_check(r)=='REJECT_AND_ROLLBACK'
    r['observed_effects'][0]['max_penetration_m']=.00001
    monkeypatch.setattr('robot_agent.contact_probe.archived_probe',lambda *a:{'attachment_flags':['active object attachment']})
    assert EvidenceGate.physical_check(r)=='REJECT_AND_ROLLBACK'
    monkeypatch.setattr('robot_agent.contact_probe.archived_probe',lambda *a:{'attachment_flags':[]})
    assert EvidenceGate.physical_check(r)=='DEFER_UNKNOWN'

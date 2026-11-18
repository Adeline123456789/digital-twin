"""Produce a fresh decision log on stdout; old records remain immutable."""
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
from robot_agent.gate import CandidateProposal,EvidenceGate
from robot_agent.legacy import digest
from robot_agent.diagnosis import json_file

if __name__=='__main__':
    active=root/'evolution/active.json'
    before=digest(active)
    for name in ('tilt','clock'):
        path=root/f'experiments/fast_slow_v2/P3.3/{name}.json'
        data=EvidenceGate().review(CandidateProposal(**json_file(path))).record()
        assert digest(active)==before
        print(json.dumps(dict(data,proposal=str(path.relative_to(root)),proposal_sha256=digest(path),
                              active_before=before,active_after=digest(active),runtime_model_calls=0)))

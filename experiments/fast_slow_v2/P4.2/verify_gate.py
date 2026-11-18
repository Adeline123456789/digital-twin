"""Re-evaluate reviewed, input-bound proposals with the current Supervisor."""
import json
import sys
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[2]
sys.path.insert(0,str(root))
from robot_agent.supervisor import Supervisor
from robot_agent.gate import CandidateProposal
from robot_agent.legacy import digest
from robot_agent.diagnosis import json_file

if __name__=='__main__':
    for name in ('tilt','clock'):
        path=out/(name+'.json')
        before=digest(root/'evolution/active.json')
        owner=Supervisor()
        d=owner.review(CandidateProposal(**json_file(path))).record()
        after=digest(root/'evolution/active.json')
        assert before==after
        print(json.dumps(dict(d,proposal_sha256=digest(path),active_before=before,active_after=after,trace=owner.trace)))

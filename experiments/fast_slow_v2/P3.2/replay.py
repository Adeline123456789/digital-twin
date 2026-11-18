"""Replay both archived retained socket1 runs and the separately grouped original controls."""
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
from robot_agent.diagnosis import EventBuilder, json_file
from robot_agent.memory_router import MemoryRouter
from robot_agent.legacy import digest


def compare():
    cases=json_file(root/'experiments/fast_slow_v2/B1/measurements.json')['cases']
    answers=[]
    router=MemoryRouter()
    for case in cases:
        path=root/case['report']
        assert digest(path)==case['sha256']
        r=EventBuilder().build(path)
        raw=json_file(path)
        answers.append(dict(episode=r.experiment_id,profile=case['profile'],target=case['target'],
                            native=case['native'],original_success=raw['task_success'],verified=raw['recording_verified'],
                            previous_parameter_deg=raw['actions'][0].get('grasp_tilt_deg',0),decision=r.status,
                            repeated_repair_veto=router.retrieve_applicable(r.first_divergence)['veto'] if r.first_divergence else [],
                            earliest_observed_s=r.first_divergence.first_observed_time if r.first_divergence else None))
    assert answers[0]['repeated_repair_veto'] and not answers[0]['original_success']
    assert answers[1]['decision']=='UNKNOWN'
    assert all(x['decision']=='NO_DIVERGENCE_OBSERVED' and x['original_success'] for x in answers[2:])
    return dict(rows=answers,retained_target1_candidate='REJECT_AND_ROLLBACK',real_cause='UNKNOWN')


if __name__=='__main__':
    print(json.dumps(compare(),indent=2))

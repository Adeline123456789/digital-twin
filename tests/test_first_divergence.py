import json
from dataclasses import replace
from robot_agent.first_divergence import event,first_divergence
from robot_agent.diagnosis import EventBuilder, ROOT
from test_observation_contract import OBS


def test_quality_and_versions_precede_order():
    e=event(OBS,'TARGET_FRAME_RESIDUAL',status='OBSERVED',first_observed_time=1.)
    assert first_divergence([e]) == e
    assert first_divergence([e,replace(e,policy_version='other')]) is None
    assert first_divergence([replace(e,units='m')]) is None


def test_frozen_episode_order_and_controls():
    cases=json.loads((ROOT/'experiments/fast_slow_v2/B1/measurements.json').read_text())['cases']
    from robot_agent.legacy import DATA
    import pytest
    if not (DATA/'self_evolution'/cases[0]['report'].split('/')[-2]/'trajectory.jsonl').exists():
        pytest.skip('External frozen dataset unavailable')
    reports=[EventBuilder().build(ROOT/c['report']) for c in cases]
    first=reports[0].first_divergence
    assert first.kind=='GRASP_RETENTION_LOSS_OBSERVED' and round(first.first_observed_time,3)==17.200
    assert round(first.confirmation_time,3)==17.267
    terminal=reports[0].events[-1]
    assert terminal.kind=='IK_UNREACHABLE' and round(terminal.timestamp_mono,2)==18.30 and terminal.first_observed_time is None
    assert reports[1].status=='UNKNOWN'
    assert all(r.task_success and r.status=='NO_DIVERGENCE_OBSERVED' for r in reports[2:])
    assert all(r.root_cause_status=='UNKNOWN' for r in reports)

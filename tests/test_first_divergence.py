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


def test_missing_verified_video_and_bad_clock_return_unknown(tmp_path):
    import h5py
    repo=tmp_path/'repo'; data=tmp_path/'data';ident='sample'
    root=repo/'experiments/cases'/ident; root.mkdir(parents=True)
    ep=data/'complete_sim/succeed'/ident
    stats=ep/'proprio_stats';stats.mkdir(parents=True)
    (ep/'videos').mkdir()
    priv=dict(frame=0,phase='lift',objects={'yellow_plug':{'position':[0,0,0]}},contacts=[])
    raw=data/'self_evolution'/ident;raw.mkdir(parents=True)
    (raw/'trajectory.jsonl').write_text(json.dumps(dict(time=1.,phase='lift',plug=[0,0,0],grip=[0,0,0])))
    (stats/'privileged.jsonl').write_text(json.dumps(priv))
    (ep/'outcome.json').write_text(json.dumps(dict(domain='sim',success=True)))
    cameras=['cam_external_rgb.mp4','cam_left_wrist_rgb.mp4','cam_right_wrist_rgb.mp4']
    (stats/'metadata.json').write_text(json.dumps(dict(domain='sim',success=True,fps=30,units={'end_effector_pos':'mm','gripper_width_m':'metre'},
                                                    source_trial='self_evolution/sample',num_frames=1)))
    (ep/'recording_sample.json').write_text(json.dumps(dict(experiment_id='check',source=ident,video_frames=dict.fromkeys(cameras,1))))
    f=root/'result.json';f.write_text(json.dumps(dict(experiment_id=ident,task_success=True,recording_verified=True,actions=[{'profile':'original'}])))
    builder=EventBuilder(data,repo)
    assert builder.build(f).events[0].kind=='SENSOR_UNAVAILABLE'
    for name in cameras:
        (ep/'videos'/name).write_bytes(b'fixture')
    with h5py.File(stats/'proprio_stats.h5','w') as h5:
        h5['sim_time']=[2.]
    assert builder.build(f).events[0].kind=='TIMESTAMP_MISMATCH'

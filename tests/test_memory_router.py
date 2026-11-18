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


def skill_case():
    from dataclasses import replace
    from robot_agent.diagnosis import json_file,ROOT
    archived=json_file(ROOT/'experiments/research_stage4/20261112T092555Z_8812d08c/result.json')
    c=dict(archived['planner_calls'][0]['applicability'],active_arms=['left','right'],
           world_model_version=OBS.world_model_version,policy_version=OBS.policy_version)
    e=event(replace(OBS,native_source_hash=c['native_sha256'],observed_value=c['residual']),
            'TARGET_FRAME_RESIDUAL',status='OBSERVED')
    return e,c


def test_scoped_skill_and_hard_rejections():
    from dataclasses import replace
    from robot_agent.memory_router import MemoryRouter
    m=MemoryRouter()
    e,c=skill_case()
    assert m.retrieve_applicable(e,c)['plan']['offset_mm'] == c['previous_offset_mm']-c['residual'][0]
    for field in [dict(source_profile='retained_final'),dict(target_index=1),dict(units='m'),
                  dict(frame_id='world'),dict(native_source_hash='c'*64),dict(sensor_provenance='real.rgb')]:
        assert not m.retrieve_applicable(replace(e,**field),c)['plan']
    for field in [dict(active_arms=['right']),dict(holder_shift_mm=4.),dict(align_contact_free=False)]:
        assert not m.retrieve_applicable(e,dict(c,**field))['plan']
    m.pins['VS002']='withdrawn'
    assert not m.retrieve_applicable(e,c)['plan']


def test_negative_is_a_veto_only_on_cited_source():
    from robot_agent.memory_router import MemoryRouter
    from dataclasses import replace
    m=MemoryRouter();r=m.negative
    e=event(replace(OBS,experiment_id=r['experiment_id'],source_profile='retained_final',target_index=1,
                    native_source_hash=r['program_version']['source_sha256']['scripts/try_plug_pick_sim.py'],
                    world_model_version=r['initial_state']['task_config_sha256'],phase='transport',units='m',
                    sensor_provenance='sim.privileged_contact',frame_id='world'),'GRASP_RETENTION_LOSS_OBSERVED')
    assert m.retrieve_applicable(e)['veto']==['grasp_tilt_minus10_as_complete_transport_repair']
    assert not m.retrieve_applicable(replace(e,source_profile='original'))['veto']

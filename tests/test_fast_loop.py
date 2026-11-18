from dataclasses import replace
from robot_agent.fast_loop import FastLoop
from test_observation_contract import OBS
from test_memory_router import skill_case


def sample(i,phase='transport'):
    return replace(OBS,phase=phase,action_id=str(i),timestamp_mono=i/30.,frame_id='world',units='m',
                   sensor_provenance='sim.privileged_contact',
                   observed_value=dict(plug=[0,0,0],grip=[0,0,.01 if phase=='lift' else .05],contacts=[]))


def test_window_and_stream_faults():
    f=FastLoop()
    assert f.update(sample(0,'lift'))['decision']=='CONTINUE'
    assert all(f.update(sample(i))['decision']=='CONTINUE' for i in (1,2))
    a=f.update(sample(3))
    assert a['decision']=='ESCALATE_SLOW' and a['event'].first_observed_frame==1
    assert a['event'].confirmation_time==3/30
    g=FastLoop()
    g.update(sample(0,'lift'))
    assert g.update(sample(4))['decision']=='UNKNOWN'


def test_fast_outcomes():
    e,c=skill_case();f=FastLoop()
    assert f.decide(e,c)['decision']=='APPLY_VERIFIED_SIM_SKILL'
    assert f.decide(replace(e,severity='danger'))['decision']=='HALT_REQUEST'
    assert f.decide(replace(e,target_index=1),c)['decision']=='ESCALATE_SLOW'
    assert f.decide(replace(e,units='unknown'))['decision']=='UNKNOWN'


def test_slow_budget_and_deduplication():
    from robot_agent.supervisor import Supervisor
    from robot_agent.first_divergence import report
    e,c=skill_case()
    r=report('episode',[replace(e,target_index=1)],False,e.evidence_refs)
    def slow(packet,tokens,seconds):
        assert tokens==20 and seconds>0 and len(packet['events'])==1
        return dict(category='WORLD_MODEL_ERROR',checks=['compare_geometry'],input_tokens=11,output_tokens=2)
    s=Supervisor(slow,max_calls=2,token_limit=20)
    assert s.diagnose(r,c)['diagnosis']=='WORLD_MODEL_ERROR'
    assert s.diagnose(r,c)['diagnosis']=='UNKNOWN' and s.calls==1
    assert Supervisor().diagnose(r,c)['diagnosis']=='UNKNOWN'
    s=Supervisor(slow,max_calls=1,token_limit=20)
    s.deadline=0
    assert s.diagnose(r,c)['diagnosis']=='UNKNOWN' and s.calls==0

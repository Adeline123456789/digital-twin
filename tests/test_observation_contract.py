from dataclasses import replace
from robot_agent.contracts import Observation, Reference
from robot_agent.timeline import aligned_samples

REF = Reference('repository','experiments/ref.json','a'*64)
OBS = Observation('episode','original','b'*64,'scene','policy','plug_insertion',3,
                  'align','1',1.,'sim_time','verified','socket','mm','sim.socket_residual',
                  'observed',[1.,0.],(REF,))


def test_quality_before_interpretation():
    assert OBS.issue() is None
    for fields in [dict(source_profile='real',clock_source='recorded_monotonic'),dict(clock_source='CAN'),
                   dict(units='degree'),dict(frame_id='world'),dict(observed_value=None),dict(native_source_hash=''),
                   dict(timestamp_mono=float('nan')),dict(evidence_refs=())]:
        assert replace(OBS, **fields).issue()
    rgb=replace(OBS,source_profile='real',clock_source='camera_exposure',sensor_provenance='real.rgb',
                frame_id='image',units='px',observability='estimated')
    assert rgb.issue() and replace(rgb, uncertainty=2.0).issue() is None
    assert replace(rgb,sensor_provenance='real.depth').issue()


def test_bad_frame_or_timestamp_is_unknown():
    row=dict(time=1.,phase='lift',plug=[1.,0.,0.],grip=[0.,0.,0.])
    extra=dict(frame=0,phase='lift',objects={'yellow_plug':{'position':row['plug']}})
    assert aligned_samples([row],[extra],[1.],30)
    assert not aligned_samples([row],[dict(extra,frame=1)],[1.],30)
    assert not aligned_samples([row],[extra],[1.1],30)

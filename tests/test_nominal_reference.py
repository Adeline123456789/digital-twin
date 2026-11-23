import pytest
from robot_agent.nominal_reference import ReferencePlayer, freeze_payload, content_hash


def payload():
    return freeze_payload([dict(time=(i + 1) / 30, phase='align', action=[0.] * 14) for i in range(3)],
                          dict(reference_episode_id='synthetic-test'))


def test_reference_checks_hash_units_clock_and_joint_target():
    value = payload()
    player = ReferencePlayer(value, content_hash(value))
    assert len(player.frames) == 3
    with pytest.raises(ValueError, match='hash'):
        ReferencePlayer(value, '0' * 64)
    value['action_units'][0] = 'deg'
    with pytest.raises(ValueError, match='SI'):
        ReferencePlayer(value, content_hash(value))
    value = payload()
    value['frames'][1]['timestep'] = 0.
    with pytest.raises(ValueError, match='discontinuous'):
        ReferencePlayer(value, content_hash(value))
    value = payload()
    value['frames'][1]['nominal_joint_target'][0] = 1.
    with pytest.raises(ValueError, match='disagrees'):
        ReferencePlayer(value, content_hash(value))

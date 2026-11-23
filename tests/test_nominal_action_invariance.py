from dataclasses import FrozenInstanceError
import pytest
from robot_agent.nominal_reference import ReferencePlayer, freeze_payload, content_hash


def test_player_owns_immutable_actions_without_live_scene_input():
    value = freeze_payload([dict(time=1 / 30, phase='insert', action=[0.] * 14)], {})
    player = ReferencePlayer(value, content_hash(value))
    action_hash = player.nominal_action_hash
    value['frames'][0]['nominal_action'][7] = .5
    assert player.command(0).nominal_action == (0.,) * 14
    assert player.nominal_action_hash == action_hash
    with pytest.raises(FrozenInstanceError):
        player.command(0).phase = 'changed'
    assert set(vars(player)) == {'frames', 'reference_hash', 'nominal_action_hash'}

import numpy as np
import pytest

from robot_agent.full_task_observation import (
    FULL_TASK_FEATURES_PER_FRAME,
    FullTaskProprioHistory,
    wrist_relative_features,
)


def test_full_task_proprio_history_has_only_deployable_inputs():
    history = FullTaskProprioHistory(15)
    observation = history.append(
        joint_position=np.zeros(6),
        joint_velocity=np.zeros(6),
        nominal_action=np.zeros(7),
        controller_target=np.zeros(7),
        gripper_width=.027,
        phase='lift',
    )

    assert observation.shape == (15 * FULL_TASK_FEATURES_PER_FRAME,)
    assert np.isfinite(observation).all()
    assert observation[-6:].tolist() == [1., 0., 0., 0., 0., 0.]

    with pytest.raises(ValueError, match='full-task phase'):
        history.append(joint_position=np.zeros(6), joint_velocity=np.zeros(6),
                       nominal_action=np.zeros(7), controller_target=np.zeros(7),
                       gripper_width=.027, phase='grasp')


def test_wrist_rgb_features_measure_housing_relative_to_two_fingers():
    rgb = np.full((120, 160, 3), 180, np.uint8)
    rgb[50:115, 10:55] = 5
    rgb[50:115, 105:150] = 5
    rgb[55:105, 67:97] = [170, 130, 10]

    centered = wrist_relative_features(rgb)
    shifted = rgb.copy()
    shifted[55:105, 67:97] = 180
    shifted[55:105, 72:102] = [170, 130, 10]
    moved = wrist_relative_features(shifted)

    assert centered.visibility == moved.visibility == 1.
    assert centered.confidence > 0.
    assert moved.relative_xy[0] > centered.relative_xy[0]
    assert centered.vector().shape == (4,)


def test_wrist_rgb_features_abstain_without_yellow_housing():
    features = wrist_relative_features(np.zeros((40, 60, 3), np.uint8))

    assert features.visibility == features.confidence == 0.
    assert features.yellow_center is None

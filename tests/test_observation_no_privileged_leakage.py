from dataclasses import fields
import numpy as np
import pytest
from robot_agent.observation_adapter import adapt, DeployableObservation, extract_rgb
from robot_agent.interaction_observer import alignment_event
from robot_agent.interaction_events import Truth


def test_adapter_accepts_only_explicit_deployable_inputs_and_unknown_propagates():
    values = dict(rgb=np.zeros((480, 640, 3), np.uint8), joint_position=[0.] * 14,
                  joint_velocity=[0.] * 14, nominal_action=[0.] * 14, phase='align', timestamp=1.)
    observation = adapt(**values)
    assert observation.visual.relative_image_residual is None
    assert alignment_event(observation, None).status is Truth.UNKNOWN
    with pytest.raises(TypeError):
        adapt(**values, simulator_success=True)
    assert {f.name for f in fields(DeployableObservation)} == {
        'joint_position', 'joint_velocity', 'command_feedback_residual', 'gripper_width',
        'phase', 'nominal_action', 'timestamp', 'visual'}


def test_rgb_color_features_are_candidates_not_privileged_labels():
    image = np.zeros((480, 640, 3), np.uint8)
    image[300:400, 250:350] = [220, 180, 10]
    features = extract_rgb(image)
    assert features.plug_housing_center == (299.5, 349.5)
    assert features.plug_tip_pixel is None
    assert features.feature_score == 0
    assert features.semantic_validation_status == 'UNVALIDATED'

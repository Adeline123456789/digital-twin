from dataclasses import replace
from unittest.mock import patch
import numpy as np
from robot_agent.observation_adapter import VisualTracker, extract_rgb, adapt
from robot_agent.interaction_observer import alignment_event
from robot_agent.interaction_events import Truth


def test_identity_survives_occlusion_without_false_alignment():
    image = np.zeros((80, 100, 3), np.uint8)
    tracker = VisualTracker()
    candidate = replace(extract_rgb(image), socket_center=(50., 30.), socket_keypoints=((50., 30.),),
                        visible_ac_panels=3, visible_usb_panels=1)
    with patch('robot_agent.observation_adapter.extract_rgb', return_value=candidate):
        first = tracker.update(image, 0., 'transport')
    assert first.target_id == 3 and first.state == 'TARGET_LOCKED'
    lost = tracker.update(image, .033, 'insert')
    assert lost.target_id == 3 and lost.state == 'OCCLUDED_CONTACT'
    assert lost.relative_image_residual is None and lost.tracking_uncertainty > 0
    prediction = dict(keypoints=((51., 30.), (50., 30.)), visible=[True, True], feature_score=.95)
    recovered = tracker.update(image, .066, 'insert', prediction)
    assert recovered.state == 'LOCAL_TRACK' and recovered.relative_image_residual == (-1., 0.)


def test_feature_score_alone_cannot_resolve_semantic_unknown():
    image = np.zeros((80, 100, 3), np.uint8)
    obs = adapt(rgb=image, joint_position=[0.] * 14, joint_velocity=[0.] * 14,
                nominal_action=[0.] * 14, phase='align', timestamp=1.)
    obs = replace(obs, visual=replace(obs.visual, feature_score=1., relative_image_residual=(0., 0.)))
    assert alignment_event(obs, None).status is Truth.UNKNOWN

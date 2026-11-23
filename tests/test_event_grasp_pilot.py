import numpy as np
import pytest
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_event_grasp_pilot import set_opening, video_grasp
from audit_event_grasp_contacts import black_robot_pair


def test_only_right_opening_changes_and_pose_is_robot_derived():
    position = np.array([0.] * 6 + [.060] + [0.] * 6 + [.060])
    expected = position.copy()
    expected[13] = .040
    set_opening(position)
    np.testing.assert_array_equal(position, expected)
    with pytest.raises(ValueError):
        set_opening(np.zeros(7))
    class IK:
        pass
    ik = IK()
    video_grasp(ik, lambda instance: setattr(instance, 'rotation', np.eye(3)))
    np.testing.assert_allclose(Rotation.from_euler('z', 90., degrees=True).as_matrix() @ ik.rotation, np.eye(3), atol=1e-15)


def test_contact_audit_counts_robot_black_pairs_only():
    assert black_robot_pair(['right/link7', 'task/black_charger'])
    assert black_robot_pair(['task/black_charger', 'right/gripper_center'])
    assert not black_robot_pair(['task/board', 'task/black_charger'])
    assert not black_robot_pair(['task/yellow_plug', 'right/link7'])

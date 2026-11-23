import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_grasp_dynamics import classify_window


def pair():
    return [dict(commanded_aperture_m=.027, actual_aperture_m=.027,
                 plug_in_gripper_position_m=[0., 0., 0.], plug_in_gripper_rotation=np.eye(3).tolist(),
                 left_finger_contact=True, right_finger_contact=True, finger_plug_geom_pairs=[], penetration_m=.00001) for _ in range(2)]


def test_command_change_and_possible_reseating_are_distinguished():
    rows = pair(); rows[1]['commanded_aperture_m'] = .026
    assert classify_window(rows)['classification'] == 'CONTROL_COMMAND_CHANGE'
    rows = pair(); rows[1].update(actual_aperture_m=.026, plug_in_gripper_position_m=[.001, 0, 0], left_finger_contact=False)
    assert classify_window(rows)['classification'] == 'POSSIBLE_SLIP_OR_RESEATING'


def test_no_visual_or_solver_causality_is_invented():
    assert classify_window(pair())['classification'] == 'UNRESOLVED_DYNAMICS'
    assert classify_window(pair(), visual_change_measured=True)['classification'] == 'VISUAL_PERSPECTIVE_ONLY'
    rows = pair(); rows[1]['penetration_m'] = .0003
    assert classify_window(rows)['classification'] == 'SIM_CONTACT_ARTIFACT_CANDIDATE'

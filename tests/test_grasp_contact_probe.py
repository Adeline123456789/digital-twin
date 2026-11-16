import numpy as np
from grasp_contact_probe import in_friction_cone


def test_point_contact_friction_cone():
    assert in_friction_cone(np.array([1., 0, 0]), np.array([2., 1, 0]), .6)
    assert not in_friction_cone(np.array([1., 0, 0]), np.array([1., 1, 0]), .6)
    assert not in_friction_cone(np.array([1., 0, 0]), np.array([-1., 0, 0]), .6)

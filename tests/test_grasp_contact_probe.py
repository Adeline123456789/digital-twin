import numpy as np
from grasp_contact_probe import in_friction_cone, probe_target


def test_negative_replays_change_only_intended_actuator_targets():
    target = np.array([90., 0., -45., 0., 20., 0., .3])
    initial = np.array([0., 0., 0., 0., 0., 0., .06])
    written = probe_target(target, initial, 'written')
    opened = probe_target(target, initial, 'open')
    frozen = probe_target(target, initial, 'frozen')
    np.testing.assert_array_equal(opened[:6], written[:6])
    assert opened[-1] == .095 and written[-1] == .3 * .095
    np.testing.assert_array_equal(frozen, initial)
    frozen[0] = 7.
    assert initial[0] == 0. and target[0] == 90.


def test_point_contact_friction_cone():
    assert in_friction_cone(np.array([1., 0, 0]), np.array([2., 1, 0]), .6)
    assert not in_friction_cone(np.array([1., 0, 0]), np.array([1., 1, 0]), .6)
    assert not in_friction_cone(np.array([1., 0, 0]), np.array([-1., 0, 0]), .6)

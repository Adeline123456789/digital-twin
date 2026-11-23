import numpy as np
from robot_agent.residual_controller import SmoothResidual, inject_xy
from robot_agent.proprio_history import ProprioHistory


def test_residual_bounds_speed_and_arm_isolation():
    smoother = SmoothResidual()
    previous = np.zeros(2)
    velocity = np.zeros(2)
    rng = np.random.default_rng(7)
    for i in range(1000):
        target = ([10., -10.] if i < 100 else [-10., 10.]) if i < 200 else rng.uniform(-10, 10, 2)
        xy = smoother.update(target)
        assert np.max(abs(xy)) <= .002
        assert np.max(abs(xy - previous)) <= .003 / 30 + 1e-12
        assert np.max(abs(smoother.velocity - velocity)) <= .03 / 30 + 1e-12
        velocity = smoother.velocity.copy()
        previous = xy
    nominal = np.arange(14) / 100
    action = inject_xy(nominal, [.001, .001], np.ones((6, 2)), -np.ones(14), np.ones(14))
    np.testing.assert_array_equal(action[:7], nominal[:7])
    np.testing.assert_allclose(action[7:13] - nominal[7:13], .002)
    assert action[13] == nominal[13]


def test_history_contains_only_right_motor_and_command_channels():
    h = ProprioHistory(15)
    value = h.append(position=[0.] * 7, velocity=[0.] * 7, nominal_action=[0.] * 7,
                     controller_target=[.01] * 7, phase='align')
    assert value.shape == (480,)
    assert np.isfinite(value).all()

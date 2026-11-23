"""Small, bounded XY correction with velocity/acceleration limits."""
import numpy as np


class SmoothResidual:
    def __init__(self, bound_m=.002, speed_m_s=.003, acceleration_m_s2=.03, dt=1 / 30):
        self.bound, self.speed, self.acceleration, self.dt = bound_m, speed_m_s, acceleration_m_s2, dt
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.omega = min(speed_m_s * np.e / (2 * bound_m),
                         np.sqrt(acceleration_m_s2 / (4 * bound_m)))

    def update(self, normalized_target):
        target = np.clip(np.asarray(normalized_target, dtype=float), -1., 1.) * self.bound
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Residual must be two finite XY targets')
        # Exact critically damped filter: positive impulse response preserves bounds,
        # with speed/acceleration limits even when the requested direction reverses.
        error = self.position - target
        transient = self.velocity + self.omega * error
        decay = np.exp(-self.omega * self.dt)
        position = target + (error + transient * self.dt) * decay
        self.velocity = (self.velocity - self.omega * transient * self.dt) * decay
        self.position = position
        return position.copy()


def inject_xy(nominal, xy_m, joint_map, lower, upper):
    action = np.asarray(nominal).copy()
    delta = np.clip(np.asarray(joint_map) @ xy_m, -.03, .03)
    action[7:13] = np.clip(action[7:13] + delta, lower[7:13], upper[7:13])
    return action

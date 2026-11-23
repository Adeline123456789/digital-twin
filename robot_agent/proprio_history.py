"""Student input boundary: right-arm motor feedback and commands only."""
from collections import deque
import numpy as np


PHASES = ('align', 'insert', 'release', 'retreat')
FEATURES_PER_FRAME = 32


class ProprioHistory:
    def __init__(self, frames=15):
        self.frames = frames
        self.values = deque(maxlen=frames)

    def append(self, *, position, velocity, nominal_action, controller_target, phase):
        q, dq, nominal, target = (np.asarray(v, dtype=float) for v in
                                  (position, velocity, nominal_action, controller_target))
        if any(v.shape != (7,) or not np.isfinite(v).all() for v in (q, dq, nominal, target)):
            raise ValueError('Expected right-arm 6 joints (rad) + gripper width (m)')
        state_scale = np.array([np.pi] * 6 + [.08])
        velocity_scale = np.array([1.] * 6 + [.08])
        residual_scale = np.array([.02] * 6 + [.005])
        value = np.r_[q / state_scale, dq / velocity_scale, nominal / state_scale,
                      (target - q) / residual_scale, [float(phase == p) for p in PHASES]].astype(np.float32)
        self.values.append(value)
        while len(self.values) < self.frames:
            self.values.appendleft(value.copy())
        return self.observation()

    def observation(self):
        return np.concatenate(self.values).astype(np.float32)

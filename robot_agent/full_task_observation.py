"""Deployable post-grasp student observations for the full-task experiment."""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


FULL_TASK_PHASES = ('lift', 'transport', 'align', 'insert', 'release', 'retreat')
FULL_TASK_FEATURES_PER_FRAME = 33


class FullTaskProprioHistory:
    """Fifteen-frame right-arm history; no object, contact or success state."""

    def __init__(self, frames=15):
        self.frames = frames
        self.values = deque(maxlen=frames)

    def append(self, *, joint_position, joint_velocity, nominal_action,
               controller_target, gripper_width, phase):
        q, dq = np.asarray(joint_position, float), np.asarray(joint_velocity, float)
        nominal, target = np.asarray(nominal_action, float), np.asarray(controller_target, float)
        gripper = float(gripper_width)
        if (q.shape != (6,) or dq.shape != (6,) or nominal.shape != (7,) or target.shape != (7,)
                or not all(np.isfinite(v).all() for v in (q, dq, nominal, target))
                or not np.isfinite(gripper) or phase not in FULL_TASK_PHASES):
            raise ValueError('Expected finite right-arm SI feedback, command and full-task phase')
        feedback = np.r_[q, gripper]
        action_scale = np.r_[[np.pi] * 6, .08]
        residual_scale = np.r_[[.02] * 6, .005]
        value = np.r_[
            q / np.pi,
            dq,
            nominal / action_scale,
            (target - feedback) / residual_scale,
            gripper / .08,
            [float(phase == name) for name in FULL_TASK_PHASES],
        ].astype(np.float32)
        self.values.append(value)
        while len(self.values) < self.frames:
            self.values.appendleft(value.copy())
        return self.observation()

    def observation(self):
        return np.concatenate(self.values).astype(np.float32)


@dataclass(frozen=True, slots=True)
class WristRelativeFeatures:
    relative_xy: tuple
    yellow_center: tuple | None
    finger_centers: tuple
    visibility: float
    confidence: float

    def vector(self):
        return np.asarray((*self.relative_xy, self.visibility, self.confidence), dtype=np.float32)


def _components(mask, min_area):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [(labels == index, stats[index], centers[index]) for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= min_area]


def wrist_relative_features(rgb):
    """Yellow-housing position relative to the two dark gripper fingers in RGB."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB uint8 image')
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow_components = _components(cv2.inRange(hsv, (15, 80, 35), (40, 255, 255)), 64)
    if not yellow_components:
        return WristRelativeFeatures((0., 0.), None, (), 0., 0.)
    yellow_mask, yellow_stats, yellow_center = max(
        yellow_components, key=lambda item: item[1][cv2.CC_STAT_AREA])
    dark = hsv[:, :, 2] < 55
    dark[:max(0, yellow_stats[cv2.CC_STAT_TOP] - 5)] = False
    dark[yellow_mask] = False
    finger_components = _components(dark, 100)
    left = [item for item in finger_components if item[2][0] < yellow_center[0]]
    right = [item for item in finger_components if item[2][0] > yellow_center[0]]
    if not left or not right:
        return WristRelativeFeatures((0., 0.), tuple(map(float, yellow_center)), (), 0., 0.)
    left_finger = max(left, key=lambda item: item[1][cv2.CC_STAT_AREA])
    right_finger = max(right, key=lambda item: item[1][cv2.CC_STAT_AREA])
    finger_centers = (tuple(map(float, left_finger[2])), tuple(map(float, right_finger[2])))
    midpoint = (np.asarray(finger_centers[0]) + finger_centers[1]) / 2
    relative = (np.asarray(yellow_center) - midpoint) / np.asarray([width, height])
    yellow_score = min(1., yellow_stats[cv2.CC_STAT_AREA] / 1000)
    finger_score = min(1., min(left_finger[1][cv2.CC_STAT_AREA],
                               right_finger[1][cv2.CC_STAT_AREA]) / 1000)
    return WristRelativeFeatures(tuple(map(float, relative)), tuple(map(float, yellow_center)),
                                 finger_centers, 1., float(yellow_score * finger_score))

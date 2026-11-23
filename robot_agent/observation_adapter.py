"""RGB/proprioception only. Pixel candidates abstain when target identity is occluded."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VisualFeatures:
    plug_tip_pixel: tuple | None
    socket_center: tuple | None
    socket_keypoints: tuple
    relative_image_residual: tuple | None
    confidence: float
    visibility: str
    plug_housing_center: tuple | None
    visible_ac_panels: int
    visible_usb_panels: int
    metal_pin_candidates: int


@dataclass(frozen=True, slots=True)
class DeployableObservation:
    joint_position: tuple
    joint_velocity: tuple
    command_feedback_residual: tuple
    gripper_width: float
    phase: str
    nominal_action: tuple
    timestamp: float
    visual: VisualFeatures


def components(mask, min_area):
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [(labels == i, stats[i], centers[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def extract_rgb(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB uint8 image')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 35), (40, 255, 255))
    housings = components(yellow, 64)
    housing = max(housings, key=lambda c: c[1][cv2.CC_STAT_AREA]) if housings else None
    blue = cv2.inRange(hsv, (95, 40, 25), (130, 255, 255))
    # Fill each blue panel's holes, then count its enclosed dark slot components.
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ac, usb = [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12 or not .5 <= w / h <= 2:
            continue
        panel = np.zeros(blue.shape, dtype=np.uint8)
        cv2.drawContours(panel, [contour], -1, 1, cv2.FILLED)
        interior = cv2.erode(panel, np.ones((3, 3), dtype=np.uint8))
        holes = components((hsv[:, :, 2] < 70) & (interior != 0), 4)
        points = tuple(tuple(map(float, center)) for _, _, center in holes)
        if len(points) == 5:
            ac.append((tuple(np.mean(points, axis=0)), points))
        elif len(points) == 2:
            usb.append((tuple(np.mean(points, axis=0)), points))
    socket, keypoints = None, ()
    # shortcut: outlet 3 requires the complete 1 USB + 3 AC row; add an RGB-only
    # temporal identity model before supporting an occluded target panel.
    if len(ac) == 3 and len(usb) == 1:
        socket, keypoints = max(ac, key=lambda p: np.linalg.norm(np.array(p[0]) - usb[0][0]))
    pins = []
    if housing is not None:
        mask, stats, center = housing
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((19, 19), np.uint8)) & ~mask
        metal = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 110) & (ring != 0)
        for pixel_mask, stat, centroid in components(metal, 6):
            x, y, w, h, area = stat
            if 1.8 <= max(w, h) / min(w, h) <= 12 and area <= stats[cv2.CC_STAT_AREA] * .1:
                yy, xx = np.nonzero(pixel_mask)
                points = np.column_stack((xx, yy))
                endpoint = points[np.argmax(np.linalg.norm(points - center, axis=1))]
                pins.append(tuple(map(float, endpoint)))
    tip = tuple(np.mean(pins, axis=0)) if len(pins) in (2, 3) else None
    # These are candidates, not semantic confidence calibrated on held-out data.
    available = tip is not None and socket is not None
    return VisualFeatures(tip, socket, keypoints,
                          tuple(np.array(socket) - tip) if available else None,
                          .5 if available else 0., 'CANDIDATE_UNVALIDATED' if available else 'UNKNOWN_OCCLUDED_OR_AMBIGUOUS',
                          tuple(map(float, housing[2])) if housing is not None else None,
                          len(ac), len(usb), len(pins))


def adapt(*, rgb, joint_position, joint_velocity, nominal_action, phase, timestamp):
    q, velocity, action = (np.asarray(value, dtype=float) for value in (joint_position, joint_velocity, nominal_action))
    if any(value.shape != (14,) or not np.isfinite(value).all() for value in (q, velocity, action)) or not np.isfinite(timestamp):
        raise ValueError('Expected finite dual-arm SI proprioception and clock')
    return DeployableObservation(tuple(q), tuple(velocity), tuple(action - q), float(q[13]), phase,
                                 tuple(action), float(timestamp), extract_rgb(rgb))

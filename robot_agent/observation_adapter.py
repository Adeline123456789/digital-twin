"""RGB/proprioception only. Pixel candidates abstain when target identity is occluded."""
from dataclasses import dataclass, replace
import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VisualFeatures:
    plug_tip_pixel: tuple | None
    socket_center: tuple | None
    socket_keypoints: tuple
    relative_image_residual: tuple | None
    feature_score: float
    semantic_validation_status: str
    visibility: str
    tracking_uncertainty: float
    plug_housing_center: tuple | None
    visible_ac_panels: int
    visible_usb_panels: int
    metal_pin_candidates: int
    target_id: int | None = None
    state: str = 'UNLOCKED'
    direct_keypoints: tuple = ()
    tracked_keypoints: tuple = ()
    last_valid_timestamp: float | None = None


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


def extract_rgb(rgb, target_hint=None):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB uint8 image')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 35), (40, 255, 255))
    housings = components(yellow, 64)
    housing = max(housings, key=lambda c: c[1][cv2.CC_STAT_AREA]) if housings else None
    blue = cv2.inRange(hsv, (95, 40, 25), (130, 255, 255))
    # Fill each blue panel's holes, then count its enclosed dark slot components.
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ac, usb, local_panels = [], [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12 or not .5 <= w / h <= 2:
            continue
        panel = np.zeros(blue.shape, dtype=np.uint8)
        cv2.drawContours(panel, [contour], -1, 1, cv2.FILLED)
        interior = cv2.erode(panel, np.ones((3, 3), dtype=np.uint8))
        holes = components((hsv[:, :, 2] < 70) & (interior != 0), 4)
        points = tuple(tuple(map(float, center)) for _, _, center in holes)
        if points:
            local_panels.append((tuple(np.mean(points, axis=0)), points))
        if len(points) == 5:
            ac.append((tuple(np.mean(points, axis=0)), points))
        elif len(points) == 2:
            usb.append((tuple(np.mean(points, axis=0)), points))
    socket, keypoints = None, ()
    # Full-board evidence is used once for identity, then retained by VisualTracker.
    if len(ac) == 3 and len(usb) == 1:
        socket, keypoints = max(ac, key=lambda p: np.linalg.norm(np.array(p[0]) - usb[0][0]))
    if target_hint is not None and local_panels:
        local = min(local_panels, key=lambda p: np.linalg.norm(np.asarray(p[0]) - target_hint))
        if np.linalg.norm(np.asarray(local[0]) - target_hint) < 25:
            socket, keypoints = local
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
    tip = min(pins, key=lambda p: p[1]) if pins else None
    socket = min(keypoints, key=lambda p: p[1]) if keypoints else None
    # shortcut: image-top candidates can confuse the two utility slots; held-out
    # semantic calibration must qualify them, otherwise use the RGB heatmap model.
    available = tip is not None and socket is not None
    return VisualFeatures(tip, socket, keypoints,
                          tuple(np.array(socket) - tip) if available else None,
                          min(1., len(pins) / 3) if available else 0., 'UNVALIDATED',
                          'VISIBLE_CANDIDATE' if available else 'UNKNOWN_OCCLUDED_OR_AMBIGUOUS', 0. if available else float('inf'),
                          tuple(map(float, housing[2])) if housing is not None else None,
                          len(ac), len(usb), len(pins))


class VisualTracker:
    """RGB-only identity memory and local optical flow; no simulation inputs."""
    def __init__(self, calibration=None, global_lock=None):
        self.calibration = calibration
        self.global_lock = global_lock
        self.target_id = None
        self.keypoints = ()
        self.last_valid_timestamp = None
        self.previous_gray = None
        self.previous_time = None
        self.uncertainty = float('inf')

    def update(self, rgb, timestamp, phase, prediction=None):
        hint = np.mean(self.keypoints, axis=0) if self.target_id is not None and self.keypoints else None
        candidate = extract_rgb(rgb, target_hint=hint)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        continuous = self.previous_time is None or 0 < timestamp - self.previous_time <= .2
        if self.target_id is None and self.global_lock is not None:
            lock = self.global_lock(rgb)
            if lock:
                self.target_id, self.keypoints = lock['target_id'], lock['keypoints']
                self.last_valid_timestamp, self.uncertainty = timestamp, 0.
        if self.target_id is None and candidate.socket_center is not None:
            self.target_id = 3
            self.keypoints = candidate.socket_keypoints
            self.last_valid_timestamp = timestamp
            self.uncertainty = 0.
        direct = (prediction['keypoints'] if prediction is not None
                  else (candidate.plug_tip_pixel, candidate.socket_center))
        visible = (prediction['visible'] if prediction is not None
                   else [point is not None for point in direct])
        score = float(prediction['feature_score']) if prediction is not None else candidate.feature_score
        tracked = ()
        if self.keypoints and self.previous_gray is not None and continuous:
            points, status, error = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray,
                np.asarray(self.keypoints, np.float32).reshape(-1, 1, 2), None)
            if status is not None and status.all():
                tracked = tuple(tuple(map(float, p)) for p in points.reshape(-1, 2))
                self.keypoints = tracked
                self.uncertainty += .2 + float(np.mean(error)) / 10
            else:
                self.uncertainty += 3.
        if not continuous:
            self.uncertainty = float('inf')
        available = self.target_id == 3 and all(visible) and continuous
        if available:
            self.keypoints = tuple(tuple(map(float, p)) for p in direct)
            self.last_valid_timestamp = timestamp
            self.uncertainty = 0.
        state = ('UNLOCKED' if self.target_id is None else 'LOCAL_TRACK' if available else
                 'OCCLUDED_CONTACT' if phase in ('align', 'insert', 'release') else 'TARGET_LOCKED')
        calibration = self.calibration
        validated = bool(calibration and calibration['accepted'] and available
                         and score >= calibration['feature_score_threshold'])
        self.previous_gray, self.previous_time = gray, timestamp
        tip, hole = direct
        return replace(candidate, plug_tip_pixel=tuple(tip) if visible[0] else None,
            socket_center=tuple(hole) if visible[1] else None,
            relative_image_residual=tuple(np.asarray(hole) - tip) if available else None,
            feature_score=score, semantic_validation_status='VALIDATED' if validated else 'UNVALIDATED',
            visibility='VISIBLE_CANDIDATE' if available else 'UNKNOWN_OCCLUDED_OR_AMBIGUOUS',
            tracking_uncertainty=self.uncertainty, target_id=self.target_id, state=state,
            direct_keypoints=tuple(tuple(p) if seen else None for p, seen in zip(direct, visible)),
            tracked_keypoints=tracked, last_valid_timestamp=self.last_valid_timestamp)


def adapt(*, rgb, joint_position, joint_velocity, nominal_action, phase, timestamp):
    q, velocity, action = (np.asarray(value, dtype=float) for value in (joint_position, joint_velocity, nominal_action))
    if any(value.shape != (14,) or not np.isfinite(value).all() for value in (q, velocity, action)) or not np.isfinite(timestamp):
        raise ValueError('Expected finite dual-arm SI proprioception and clock')
    return DeployableObservation(tuple(q), tuple(velocity), tuple(action - q), float(q[13]), phase,
                                 tuple(action), float(timestamp), extract_rgb(rgb))

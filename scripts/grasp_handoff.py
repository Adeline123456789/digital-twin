"""Reject observed in-hand changes before transport; never certify insertion or grip stability."""
import math


def yellow_top(image):
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 90, 60]), np.array([45, 255, 255]))
    # shortcut: central wrist ROI for this plug; revise if the camera mount changes.
    mask[:, :180] = 0
    mask[:, 470:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 300:
        return None
    return cv2.boundingRect(max(contours, key=cv2.contourArea))[1]


def screen_lift(grasp_width_mm, lift_width_mm, grasp_yellow_top_px, lift_yellow_top_px):
    # shortcut: this fixed wrist mount/plug only; recalibrate thresholds if either changes.
    values = (grasp_width_mm, lift_width_mm, grasp_yellow_top_px, lift_yellow_top_px)
    if any(v is None or not math.isfinite(v) for v in values):
        return dict(allow_transport=False, reason='Missing grasp/lift observation')
    width_change = abs(lift_width_mm - grasp_width_mm)
    image_change = abs(lift_yellow_top_px - grasp_yellow_top_px)
    allowed = width_change <= 5. and image_change <= 15.
    return dict(allow_transport=allowed,
        reason='No screened change detected' if allowed else 'Grasp-to-lift observation changed',
        encoder_width_change_mm=width_change, yellow_top_change_px=image_change,
        thresholds=dict(encoder_width_change_mm=5., yellow_top_change_px=15.),
        limit='Change rejection only: an already tilted grasp can pass. Not absolute pose, force or stable-grasp certification')

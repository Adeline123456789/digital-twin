"""One-time RGB board-layout identity for the current external camera view."""
from itertools import combinations
import cv2
import numpy as np


def lock_outlet_three(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (95, 40, 25), (130, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 12 and h >= 12 and .5 <= w / h <= 2:
            panels.append((np.array([x + w / 2, y + h / 2]), w, h))
    for group in combinations(panels, 4):
        ordered = sorted(group, key=lambda p: p[0][0])
        centers = np.array([p[0] for p in ordered])
        widths = np.array([p[1] for p in ordered])
        gaps = np.diff(centers[:, 0])
        if (widths.max() / widths.min() < 1.4 and np.ptp(centers[:, 1]) < widths.mean() * .3
                and np.all((gaps > widths.mean() * .9) & (gaps < widths.mean() * 1.8))):
            # shortcut: this task's external view orders USB, AC1, AC2, AC3 left
            # to right; a reversed/unseen board orientation needs revalidation.
            return dict(target_id=3, keypoints=(tuple(centers[-1]),), evidence='four collinear equal-spaced blue panels; task layout order')
    return None

import numpy as np
import cv2
from robot_agent.target_lock import lock_outlet_three


def test_four_panel_layout_locks_once_and_incomplete_layout_does_not():
    rgb = np.zeros((100, 200, 3), np.uint8)
    blue = cv2.cvtColor(np.uint8([[[110, 180, 180]]]), cv2.COLOR_HSV2RGB)[0, 0]
    for x in (20, 45, 70, 95): rgb[40:60, x:x+20] = blue
    assert lock_outlet_three(rgb)['keypoints'] == ((105., 50.),)
    rgb[:, 95:] = 0
    assert lock_outlet_three(rgb) is None

import sys
from pathlib import Path
import unittest
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_rgb_motion import track_step, relative_span


class RGBMotionTests(unittest.TestCase):
    def test_known_translation_and_common_motion_cancellation(self):
        rng = np.random.default_rng(4)
        image = cv2.GaussianBlur(rng.integers(0, 256, (480, 640), np.uint8), (5, 5), 0)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (640, 480))
        points = np.float32([[192, 144], [256, 192], [384, 144], [448, 192]])
        forward, valid, error = track_step(image, moved, points)
        self.assertTrue(valid.all()); self.assertLess(error.max(), .1)
        np.testing.assert_allclose(forward-points, np.tile([2, 1], (4, 1)), atol=.1)
        tracks = np.array([points, forward]); labels = np.array([0, 0, 1, 1])
        np.testing.assert_allclose(relative_span(tracks, labels, valid), [0, 0], atol=.1)
        tracks[1, labels == 0, 0] += 3
        np.testing.assert_allclose(relative_span(tracks, labels, valid), [0, -3], atol=.1)

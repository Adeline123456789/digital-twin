import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from observe_real_failures import yellow_top
class VisualObservationTests(unittest.TestCase):
    def test_visible_yellow_motion_and_missing_object(self):
        im=np.zeros((480,640,3),dtype=np.uint8)
        self.assertIsNone(yellow_top(im,85))
        im[320:450,240:400]=(0,180,180)
        self.assertEqual(yellow_top(im,85),320)
        shifted=np.zeros_like(im);shifted[330:460,240:400]=(0,180,180)
        self.assertEqual(yellow_top(shifted,85),330)

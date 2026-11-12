import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_rgbd_geometry import mask_distance,finger_surfaces


class GeometryTests(unittest.TestCase):
    def test_out_of_image_points_are_counted(self):
        mask=np.zeros((20,20),bool);mask[5:15,5:15]=True
        r=mask_distance(np.array([[10,10],[30,10]]),mask)
        self.assertEqual(r['in_image_fraction'],.5)
        self.assertEqual(r['median_px'],0)

    def test_finger_points_are_inside_dark_surfaces(self):
        image=np.full((480,640,3),255,np.uint8)
        image[380:450,150:200]=0;image[380:450,480:530]=0
        points=finger_surfaces(image)
        self.assertTrue(150<points[0][0]<200 and 480<points[1][0]<530)
        self.assertTrue(all(p[1]>=383 for p in points))

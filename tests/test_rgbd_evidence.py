import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_rgbd_evidence import depth_stats,yellow_mask


class DepthEvidenceTests(unittest.TestCase):
    def test_missing_depth_is_unknown_not_zero_distance(self):
        d=np.zeros((10,10),np.uint16);mask=np.ones(d.shape,bool)
        r=depth_stats(d,mask,.001)
        self.assertEqual(r['positive_fraction'],0)
        self.assertIsNone(r['depth_p05_median_p95_m'])
        d[0,0]=200
        r=depth_stats(d,mask,.001)
        self.assertEqual(r['depth_p05_median_p95_m'],[.2,.2,.2])
        self.assertEqual(r['positive_fraction'],.01)

    def test_background_is_not_a_yellow_observation(self):
        self.assertFalse(yellow_mask(np.zeros((10,10,3),np.uint8)).any())

    def test_roi_excludes_larger_yellow_distractor(self):
        image=np.zeros((30,60,3),np.uint8)
        image[2:28,2:28]=[0,255,255];image[10:20,40:50]=[0,255,255]
        mask=yellow_mask(image,(35,5,55,25))
        self.assertFalse(mask[:,:30].any())
        self.assertTrue(mask[:,35:].any())

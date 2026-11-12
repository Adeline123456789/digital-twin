import sys
from pathlib import Path
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_rgb_homography import candidates


class HomographyAuditTests(unittest.TestCase):
    def test_retains_ambiguity_and_does_not_invent_metric_translation(self):
        k=np.array([[600.,0,320],[0,600,240],[0,0,1]])
        r=Rotation.from_euler('xyz',[2,4,1],degrees=True).as_matrix()
        h=k@(r+np.outer([.08,.02,.03],[0,0,1]))@np.linalg.inv(k)
        points=np.float32([[250,200],[350,200],[350,280],[250,280]])
        after=cv2.perspectiveTransform(points[:,None,:],h)[:,0]
        angle=np.rad2deg(Rotation.from_matrix(r).magnitude())
        report=candidates(h,k,points,after,angle)
        self.assertEqual(report['candidate_count'],4)
        self.assertGreaterEqual(report['visible_candidate_count'],2)
        self.assertLess(min(x['robot_angle_difference_deg'] for x in report['candidates']),1e-7)
        self.assertIsNone(report['selected_candidate'])
        self.assertFalse(report['metric_translation_available'])

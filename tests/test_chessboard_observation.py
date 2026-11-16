import sys
from pathlib import Path
import unittest
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from chessboard_observation import board_points,pose_candidates,observe


class BoardTests(unittest.TestCase):
    def test_metric_grid_and_pose_recovery(self):
        points=board_points()
        self.assertEqual(points.shape,(88,3))
        np.testing.assert_allclose(points[1]-points[0],[.02,0,0])
        K=np.array([[600.,0,320],[0,600,240],[0,0,1.]])
        r=np.array([.2,-.1,.05]);t=np.array([.02,-.01,.5])
        uv=cv2.projectPoints(points,r,t,K,np.zeros(5))[0]
        candidates=pose_candidates(uv,K,np.zeros(5))
        self.assertEqual(len(candidates),2)
        np.testing.assert_allclose(np.array(candidates[0]['T_camera_target'])[:3,3],t,atol=1e-8)
        self.assertLess(candidates[0]['reprojection_rms_px'],1e-6)

    def test_real_detector_counts_internal_corners_and_rejects_blank(self):
        image=np.full((480,640,3),255,np.uint8)
        for y in range(9):
            for x in range(12):
                if (x+y)%2==0:image[60+y*30:60+(y+1)*30,100+x*30:100+(x+1)*30]=0
        camera={'K':[[600,0,320],[0,600,240],[0,0,1]],'distortion_coeffs':[0]*5}
        self.assertEqual(observe(image,camera)['corner_count'],88)
        self.assertEqual(observe(np.full_like(image,255),camera)['status'],'TARGET_NOT_FULLY_VISIBLE')


if __name__=='__main__':unittest.main()

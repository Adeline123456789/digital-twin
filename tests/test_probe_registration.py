"""Coordinate convention and metric recovery, independent synthetic geometry."""
import sys
from pathlib import Path
import unittest
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_probe_registration import project,triangulate,fit,lk_matches


class RegistrationTests(unittest.TestCase):
    def test_lk_forward_backward_on_known_image_translation(self):
        rng=np.random.default_rng(12)
        first=cv2.GaussianBlur(rng.integers(0,256,(480,640),dtype=np.uint8),(5,5),0)
        second=cv2.warpAffine(first,np.float32([[1,0,5],[0,1,-3]]),(640,480))
        points=np.float32([[200,100],[300,200],[400,300]]).reshape(-1,1,2)
        target,good=lk_matches(first,second,points)
        self.assertTrue(np.all(good))
        np.testing.assert_allclose((target-points).reshape(-1,2),np.tile([5,-3],(3,1)),atol=.1)

    def test_recover_known_mount_with_fixed_metric_robot_motion(self):
        rng=np.random.default_rng(6)
        poses=np.repeat(np.eye(4)[None],8,axis=0)
        poses[:,:3,:3]=Rotation.from_rotvec(rng.uniform(-.2,.2,(8,3))).as_matrix()
        poses[:,:3,3]=rng.uniform(-.03,.03,(8,3))
        K=np.array([[600,0,320],[0,600,240],[0,0,1]])
        truth=np.eye(4);truth[:3,:3]=Rotation.from_euler('xyz',[2,-1,3],degrees=True).as_matrix()
        truth[:3,3]=[.01,-.02,.03]
        points=rng.uniform([-.04,-.04,.25],[.04,.04,.45],(18,3))
        vi=np.tile(np.arange(8),18);pi=np.repeat(np.arange(18),8)
        uv,z=project(points,poses,truth,K,vi,pi)
        self.assertTrue(np.all(z>0))
        observations=[{j:uv[i*8+j] for j in range(8)} for i in range(18)]
        np.testing.assert_allclose(triangulate(poses,truth,K,observations),points,atol=1e-10)
        initial=truth.copy();initial[:3,3]+=[.003,-.002,.004]
        initial[:3,:3]=Rotation.from_euler('x',1,degrees=True).as_matrix()@truth[:3,:3]
        result,(estimate,reconstruction)=fit(points,poses,initial,K,vi,pi,uv,True)
        self.assertTrue(result.success)
        np.testing.assert_allclose(estimate,truth,atol=1e-5)
        np.testing.assert_allclose(reconstruction,points,atol=1e-5)


if __name__=='__main__':unittest.main()

from pathlib import Path
import sys
import unittest
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rgb_correspondences import match_plane, refine_plane, observed_support


class RGBCorrespondenceTests(unittest.TestCase):
    def test_local_tracks_do_not_mark_distant_pixels_as_observed(self):
        roi=np.full((100,200),255,np.uint8);roi[30:40,30:40]=0
        tracks=np.float32([[20,20],[60,20],[60,80],[20,80]])
        support=observed_support(roi,tracks)
        self.assertEqual(support[50,40],255)
        self.assertEqual(support[50,150],0)
        self.assertEqual(support[35,35],0)

    def test_recovers_known_projective_warp(self):
        rng=np.random.default_rng(6)
        reference=cv2.GaussianBlur(rng.integers(0,256,(480,640),np.uint8),(3,3),0)
        h=np.array([[1.01,.025,9],[-.015,.98,12],[.00005,-.00003,1]])
        target=cv2.warpPerspective(reference,h,(640,480))
        mask=np.zeros_like(reference);mask[80:400,80:560]=255
        report,points=match_plane(reference,target,mask)
        self.assertEqual(report['status'],'IMAGE_CORRESPONDENCES_ONLY')
        self.assertGreater(report['inlier_count'],30)
        corners=np.float32([[[100,100],[500,100],[500,350],[100,350]]])
        estimated=cv2.perspectiveTransform(corners,np.array(points['homography']))
        actual=cv2.perspectiveTransform(corners,h)
        self.assertLess(np.max(np.linalg.norm(estimated-actual,axis=2)),.3)

    def test_textureless_input_has_no_fabricated_correspondences(self):
        image=np.full((480,640),128,np.uint8)
        report,points=match_plane(image,image,np.full_like(image,255))
        self.assertEqual(report['status'],'INSUFFICIENT_MATCHES')
        self.assertEqual(points,{})

    def test_ecc_refines_known_warp_with_target_mask(self):
        rng=np.random.default_rng(9)
        ref=cv2.GaussianBlur(rng.integers(30,200,(180,240),np.uint8),(5,5),0)
        truth=np.array([[1.01,.015,4],[-.01,.99,3],[.00003,-.00002,1.]])
        target=cv2.warpPerspective(ref,truth,(240,180))
        target=np.uint8(np.clip(target.astype(float)*.9+10,0,255))
        mask=np.zeros_like(ref);mask[30:150,30:210]=255
        initial=truth.copy();initial[0,2]+=1.5;initial[1,2]-=1.
        fitted,score=refine_plane(ref,target,mask,initial)
        points=np.float32([[[45,45],[190,45],[190,135],[45,135]]])
        error=np.linalg.norm(cv2.perspectiveTransform(points,fitted)-
                             cv2.perspectiveTransform(points,truth),axis=2)
        self.assertLess(error.max(),.3)
        self.assertGreater(score,.95)

    def test_ecc_uniform_image_has_no_valid_estimate(self):
        blank=np.full((80,100),128,np.uint8)
        with self.assertRaises(cv2.error):
            refine_plane(blank,blank,np.full_like(blank,255),np.eye(3))

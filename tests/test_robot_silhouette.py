import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_robot_silhouette import robot_mask, changed_pose, mask_metrics


class SilhouetteTests(unittest.TestCase):
    def test_excludes_disconnected_fixture_and_preserves_robot(self):
        image=np.full((480,640,3),180,np.uint8)
        image[200:300,500:640]=30; image[250:275,300:350]=20
        mask,roi=robot_mask(image,90)
        self.assertEqual(mask.sum(),100*140-45*70)
        self.assertFalse(mask[260,520])
        self.assertEqual(mask_metrics(mask,mask,roi)['iou'],1.)
        shifted=np.roll(mask,-30,axis=1)
        self.assertLess(mask_metrics(shifted,mask,roi)['iou'],.8)
        with self.assertRaises(ValueError): robot_mask(np.full_like(image,180),90)

    def test_bounded_pose_units_and_rotation(self):
        initial=np.eye(4)
        actual=changed_pose(initial,[0,0,1,1,0,0])
        np.testing.assert_allclose(actual[:3,3],[.03,0,0])
        np.testing.assert_allclose(actual[:3,:3]@actual[:3,:3].T,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(actual[0,0],np.cos(np.deg2rad(5)))
        np.testing.assert_array_equal(initial,np.eye(4))


if __name__=='__main__': unittest.main()

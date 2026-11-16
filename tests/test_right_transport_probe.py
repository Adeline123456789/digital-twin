import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_transport_ik import right_joint_seed


class RightSeedTests(unittest.TestCase):
    def test_right_and_dual_actions_select_the_same_named_joints(self):
        right=[.1,.2,.3,.4,.5,.6,.06]
        np.testing.assert_array_equal(right_joint_seed(right,('right',)),right[:6])
        np.testing.assert_array_equal(right_joint_seed([9]*7+right,('left','right')),right[:6])
        with self.assertRaises(ValueError):right_joint_seed(right,('left','right'))
        with self.assertRaises(ValueError):right_joint_seed(right,('left',))


if __name__=='__main__':unittest.main()

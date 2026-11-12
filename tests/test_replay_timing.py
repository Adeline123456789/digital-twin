import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_replay_timing import fit_delay
from audit_aperture_mapping import plateau_mask


class DelayTests(unittest.TestCase):
    def test_plateau_excludes_lift_and_early_grasp(self):
        t=np.arange(0,3,.25);phases=np.array(['grasp']*8+['lift']*4)
        np.testing.assert_array_equal(np.flatnonzero(plateau_mask(t,phases)),[5,6,7])

    def test_known_delay_recovery(self):
        st=np.linspace(0,3,301);prediction=60-10*st
        t=np.linspace(.4,2.5,100);observed=np.interp(t-.08,st,prediction)
        result=fit_delay(t,observed,st,prediction)
        self.assertAlmostEqual(result.x[0],.08,places=6)

    def test_plateau_mismatch_is_not_repaired_by_delay(self):
        st=np.linspace(0,3,301);t=np.linspace(.4,2.5,100)
        result=fit_delay(t,np.full(100,50.),st,np.full(301,40.))
        self.assertAlmostEqual(np.sqrt(np.mean(result.fun**2)),10.)

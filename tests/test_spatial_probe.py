import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from spatial_probe import make_reference, load_probe
from learn import digest


class SpatialProbeTests(unittest.TestCase):
    def test_open_gripper_speed_and_fold(self):
        t,q,ph=make_reference([0,0,0,0,23,0],[{'name':'holder','q_deg':[0,75,-55,0,55,-100]}])
        self.assertLessEqual(np.abs(np.diff(q[:,:6],axis=0)/np.diff(t)[:,None]).max(),12+1e-6)
        np.testing.assert_allclose(q[:,6],.060/.095)
        np.testing.assert_allclose(q[-1,:6],0)
        self.assertTrue(ph[-1].startswith('fold'))

    def test_rejects_unvalidated_or_changed_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);np.savez(p/'reference.npz',time=[0,1],targets=np.zeros((2,7)),phases=['entry','fold'])
            identity=digest(p/'reference.npz')
            (p/'manifest.json').write_text(json.dumps({'purpose':'noncontact_spatial_probe','trajectory_id':identity}))
            (p/'validation.json').write_text(json.dumps({'reference_sha256':identity,'probe_valid':False,'all_substeps_valid':True,'no_robot_task_contact':True}))
            with self.assertRaises(ValueError):load_probe(p)
            (p/'validation.json').write_text(json.dumps({'reference_sha256':identity,'probe_valid':True,'all_substeps_valid':True,'no_robot_task_contact':True}))
            load_probe(p)
            with (p/'reference.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):load_probe(p)


if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import sys
import tempfile
import os
import time
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from spatial_probe import make_reference, load_probe, require_fresh_rgb
from learn import digest


class SpatialProbeTests(unittest.TestCase):
    def test_ready_marker_does_not_replace_fresh_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            for camera in ('external','right_wrist'):
                (p/camera).mkdir();(p/camera/'READY').write_text('ready')
            with self.assertRaises(RuntimeError):require_fresh_rgb(p)
            for camera in ('external','right_wrist'):
                for name in ('0000.jpg','0001.jpg'):(p/camera/name).touch()
            require_fresh_rgb(p)
            stale=time.time()-1;os.utime(p/'right_wrist/0000.jpg',(stale,stale))
            with self.assertRaises(RuntimeError):require_fresh_rgb(p)

    def test_open_gripper_speed_and_fold(self):
        t,q,ph=make_reference([0,0,0,0,23,0],[{'name':'holder','q_deg':[0,75,-55,0,55,-100]}])
        self.assertLessEqual(np.abs(np.diff(q[:,:6],axis=0)/np.diff(t)[:,None]).max(),12+1e-6)
        np.testing.assert_allclose(q[:,6],.060/.095)
        np.testing.assert_allclose(q[-1,:6],0)
        self.assertTrue(ph[-1].startswith('fold'))

    def test_explicit_views_do_not_add_unspecified_joint_probes(self):
        views=[{'name':'holder','q_deg':[0,75,-55,0,55,-100],'center_only':True}]
        t,q,ph=make_reference([0,0,0,0,23,0],views)
        self.assertEqual(set(ph),{'entry','holder_center','holder_center_hold',
                                 'holder_return','holder_return_hold','fold','fold_hold'})
        self.assertLessEqual(np.abs(np.diff(q[:,:6],axis=0)/np.diff(t)[:,None]).max(),12+1e-6)
        np.testing.assert_allclose(q[:,6],.060/.095)

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

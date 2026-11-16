from pathlib import Path
import sys
import unittest
import json
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from trajectory_contract import RIGHT,right_targets,resample,load_reference,sha

class ContractTests(unittest.TestCase):
    def test_named_mapping_ignores_column_order_and_preserves_commands(self):
        names=list(reversed(RIGHT));actions=np.arange(7)*.001
        rows=[dict(time=0.,action=actions[::-1],phase='a'),dict(time=.1,action=(actions+.001)[::-1],phase='b')]
        q=right_targets(rows,names)
        np.testing.assert_allclose(q[0,:6],np.rad2deg(actions[:6]))
        self.assertAlmostEqual(q[0,6],actions[6]/.095)
        t,targets,phase=resample(rows,names)
        self.assertAlmostEqual(t[-1],.1)
        np.testing.assert_array_equal(targets[0],q[0]);np.testing.assert_array_equal(targets[-1],q[-1])
        self.assertEqual(phase[-1],'b')
    def test_no_silent_retiming_or_missing_names(self):
        rows=[dict(time=0,action=[0]*6+[.06],phase='a'),dict(time=.1,action=[1]+[0]*5+[.06],phase='b')]
        with self.assertRaises(ValueError):resample(rows,RIGHT)
        with self.assertRaises(ValueError):right_targets(rows,RIGHT[:-1])
    def test_execution_requires_same_reference_and_substep_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            np.savez(path/'reference.npz',time=[0,.1],targets=np.zeros((2,7)),phases=['a','b'])
            identity=sha(path/'reference.npz')
            (path/'manifest.json').write_text(json.dumps({'trajectory_id':identity}))
            for validated_hash,valid in [('another trajectory',True),(identity,False)]:
                (path/'validation.json').write_text(json.dumps(dict(reference_sha256=validated_hash,task_success=True,all_substeps_valid=valid)))
                with self.assertRaises(ValueError):load_reference(path,require_validation=True)
            (path/'validation.json').write_text(json.dumps(dict(reference_sha256=identity,task_success=True,all_substeps_valid=True)))
            self.assertEqual(load_reference(path,require_validation=True)[2],['a','b'])

if __name__=='__main__':unittest.main()

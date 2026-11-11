import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import learn
class LearningTests(unittest.TestCase):
    def test_repeat_and_unknown(self):
        self.assertEqual(learn.decide('plug_not_seated','hold_gripper_during_retreat'),'REJECT_REPEAT_REQUIRE_NEW_EVIDENCE')
        self.assertEqual(learn.decide('new_task','random_guess'),'UNKNOWN_TASK_MODE')
        self.assertEqual(learn.decide('plug_not_seated','measure_relative_pose'),'ALLOW_SIMULATION')
    def test_limit(self): self.assertLessEqual(len(learn.retrieve('plug_not_seated')),5)
    def test_exception_persisted(self):
        original=learn.SIM
        try:
            learn.SIM=Path('/nonexistent')
            # Missing provenance is an error, never a fabricated success.
            with self.assertRaises(Exception): learn.version()
        finally: learn.SIM=original
if __name__=='__main__':unittest.main()

class FailureAndRollbackTests(unittest.TestCase):
    def test_error_report_survives_missing_runtime(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'protocol').mkdir();(root/'protocol/robot_problem_solving.md').write_text('test')
            with patch.object(learn,'ROOT',root),patch.object(learn,'SIM',root/'missing'),patch.object(learn,'DATA',root/'data'):
                out,r=learn.simulate('evaluation',0,0,'missing simulator')
                self.assertEqual(r['result'],'infrastructure_error')
                self.assertTrue((out/'result.json').exists())
                self.assertTrue((out/'REPORT.md').exists())
                self.assertIsNone(r['task_success'])
    def test_regression_not_accepted(self):
        from evolve import accepts
        self.assertFalse(accepts({'v1':1,'v2':.5},[{'v2':'RETRY','expected':'UNKNOWN'}]))
        self.assertFalse(accepts({'v1':.5,'v2':.8},[{'v2':'RETRY','expected':'UNKNOWN'}]))

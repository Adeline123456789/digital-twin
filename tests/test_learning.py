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
    def test_excluded_depth_memory_cannot_authorize_a_repair(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'memory').mkdir()
            def memory(name,status,repairs):
                (root/'memory'/name).write_text(json.dumps(dict(id=name,signatures=['test'],
                    validation_status=status,ineffective_repairs=repairs)))
            memory('a.json','EXCLUDED_AS_REAL_DEPTH_USER_CONFIRMED_RGB_ONLY',[])
            with patch.object(learn,'ROOT',root):
                self.assertEqual(learn.retrieve('test'),[])
                self.assertEqual(learn.decide('test','guess','v2'),'UNKNOWN_TASK_MODE')
                memory('b.json','HYPOTHESES_TESTED_REJECTED_FOR_GLOBAL_CALIBRATION',['bad_fix'])
                self.assertEqual([x['id'] for x in learn.retrieve('test',limit=1)],['b.json'])
                self.assertEqual(learn.decide('test','bad_fix','v2'),'REJECT_REPEAT_REQUIRE_NEW_EVIDENCE')
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

class ActualRollbackGateTests(unittest.TestCase):
    def test_regression_preserves_active_version_file(self):
        import evolve
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'protocol').mkdir();(root/'protocol/robot_problem_solving.md').write_text('test')
            (root/'evolution').mkdir();prior={'version':'v2','previous':'v1','evidence':'existing'}
            (root/'evolution/active.json').write_text(json.dumps(prior))
            expected={(s,r):w for s,r,w in evolve.CASES}
            def fake_decide(s,r,v):return expected[s,r] if v=='v1' else 'RETRY'
            (root/'scripts').mkdir();(root/'scripts/learn.py').write_text('test snapshot')
            with patch.object(learn,'ROOT',root),patch.object(evolve,'ROOT',root),patch.object(evolve,'decide',side_effect=fake_decide):evolve.main()
            self.assertEqual(json.loads((root/'evolution/active.json').read_text()),prior)
            report=json.loads(next((root/'experiments/evaluation').glob('*/result.json')).read_text())
            self.assertEqual(report['validation_status'],'ROLLED_BACK')

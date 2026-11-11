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

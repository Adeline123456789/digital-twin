import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from agent_loop import validate_plan
class PlanTests(unittest.TestCase):
    def test_only_finite_bounded_parameter(self):
        for x in [float('nan'),float('inf'),5,True,'0']:
            with self.assertRaises(ValueError):validate_plan({'offset_mm':x,'hypotheses':[]})
        self.assertEqual(validate_plan({'offset_mm':0,'hypotheses':[]}),0)
    def test_no_arbitrary_commands_or_unbounded_hypotheses(self):
        with self.assertRaises(ValueError):validate_plan({'offset_mm':0,'hypotheses':[],'command':'hardware'})
        with self.assertRaises(ValueError):validate_plan({'offset_mm':0,'hypotheses':['a']*4})

class RetainedProfileTests(unittest.TestCase):
    def test_documented_retained_timing_is_passed_to_existing_script(self):
        import tempfile
        import os
        from unittest.mock import patch
        from types import SimpleNamespace
        import learn
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);sim=root/'sim';(sim/'configs/tasks').mkdir(parents=True)
            (sim/'configs/tasks/real_connector_final.yaml').write_text('holder_xy_m: [0.3, -0.3]\n')
            (root/'protocol').mkdir();(root/'protocol/robot_problem_solving.md').write_text('test')
            calls=[]
            def fake_run(cmd,**kw):
                calls.append(cmd)
                return SimpleNamespace(stdout='',returncode=1)
            with patch.object(learn,'ROOT',root),patch.object(learn,'SIM',sim),patch.object(learn,'DATA',root/'data'),patch.dict(os.environ,{'PIPER_SIM_PROFILE':'retained_final'}),patch.object(learn.subprocess,'run',side_effect=fake_run):
                learn.simulate('evaluation',0,.25,'command fidelity check')
            command=next(c for c in calls if '--grasp-roll-deg' in c)
            self.assertEqual(command[command.index('--speed')+1],'25')
            self.assertEqual(command[command.index('--grasp-height')+1],'.012')
            self.assertEqual(command[command.index('--align-corrections')+1],'4')

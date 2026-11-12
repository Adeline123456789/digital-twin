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
            (sim/'configs/tasks/real_connector_final.yaml').write_text('holder_xy_m: [0.3, -0.3]\ntarget_index: 3\n')
            (root/'protocol').mkdir();(root/'protocol/robot_problem_solving.md').write_text('test')
            calls=[]
            def fake_run(cmd,**kw):
                calls.append(cmd)
                return SimpleNamespace(stdout='',returncode=1)
            with patch.object(learn,'ROOT',root),patch.object(learn,'SIM',sim),patch.object(learn,'DATA',root/'data'),patch.dict(os.environ,{'PIPER_SIM_PROFILE':'retained_final'}),patch.object(learn.subprocess,'run',side_effect=fake_run):
                learn.simulate('evaluation',0,.25,'command fidelity check')
                learn.simulate('evaluation',0,.25,'tilt command fidelity check',grasp_tilt_deg=5.)
                learn.simulate('evaluation',0,.25,'invalid tilt check',grasp_tilt_deg=11.)
            command=next(c for c in calls if '--grasp-roll-deg' in c)
            self.assertEqual(command[command.index('--speed')+1],'25')
            self.assertEqual(command[command.index('--grasp-height')+1],'.012')
            self.assertEqual(command[command.index('--align-corrections')+1],'4')
            modified=[c for c in calls if '--grasp-tilt-deg' in c]
            self.assertEqual(len(modified),1)
            self.assertEqual(modified[0][modified[0].index('--grasp-tilt-deg')+1],'5.0')
            self.assertEqual(modified[0][modified[0].index('--speed')+1],'25')
            self.assertEqual(modified[0][modified[0].index('--grasp-height')+1],'.012')

class FeedbackMemoryTests(unittest.TestCase):
    def test_transport_failure_retrieves_previous_ineffective_repair(self):
        from learn import failure_memory
        memories=failure_memory([{'last_phase':'transport','error':'Unreachable pose'}])
        self.assertTrue(any(m['id']=='NE004' for m in memories))
        self.assertLessEqual(len(memories),5)

class ApplicabilityTests(unittest.TestCase):
    def test_wrong_timing_experience_excluded_when_actual_speed_matches_documentation(self):
        from learn import failure_memory
        memories=failure_memory([{'last_phase':'transport','error':'Unreachable pose','speed_deg_s':25}])
        self.assertFalse(any(m['id']=='NE004' for m in memories))

class SeedScopeTests(unittest.TestCase):
    def test_seed_matches_both_profile_and_target(self):
        from agent_loop import compatible_seed_history
        self.assertTrue(compatible_seed_history('original',3))
        self.assertFalse(compatible_seed_history('retained_final',3))
        self.assertFalse(compatible_seed_history('original',1))

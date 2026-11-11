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

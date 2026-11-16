import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import agent_loop
import benchmark_collection

class RightAgentTests(unittest.TestCase):
    def test_right_agent_excludes_dual_history_and_calls_right_collector(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for d in ['evolution','protocol','experiments/baseline/20261111T083720Z_f64309a8']:(root/d).mkdir(parents=True)
            (root/'evolution/active.json').write_text('{"planner_policy":"compiled_alignment_v1"}')
            (root/'protocol/robot_problem_solving.md').write_text('generic instruction')
            (root/'experiments/baseline/20261111T083720Z_f64309a8/result.json').write_text('{"experiment_id":"old-dual","actions":["DUAL_ONLY_SECRET"],"observed_effects":[]}')
            call={'answer':{'offset_mm':0,'hypotheses':[]},'tool_events':0,'usage':{'input_tokens':1},'planner_kind':'llm'}
            trial={'experiment_id':'fixture','task_success':True,'result':'success','observed_effects':[], 'execution_time':1,'collection_component_seconds':1,'recording_verified':True}
            def collect(*a,**k):
                self.assertEqual(agent_loop.os.environ['PIPER_ACTIVE_ARMS'],'right');return trial
            with patch.object(agent_loop,'ROOT',root),patch.object(agent_loop,'record',return_value=(root,{'experiment_id':'agent'})),patch.object(agent_loop,'finish'),patch.object(agent_loop,'plan_with_memory',return_value=call) as planner,patch.object(benchmark_collection,'run_case',side_effect=collect),patch.dict(agent_loop.os.environ,{'PIPER_ACTIVE_ARMS':'right','PIPER_SIM_PROFILE':'original','PIPER_TARGET_INDEX':'3'}):
                result=agent_loop.run('D',0,publish=False)
            self.assertEqual(result['actual_planner_policy'],'llm')
            self.assertTrue(result['initial_state']['incompatible_history_excluded'])
            self.assertNotIn('DUAL_ONLY_SECRET',planner.call_args.args[0])
            self.assertFalse(planner.call_args.kwargs['include_catalog'])
            self.assertTrue(result['task_success'])

    def test_connection_rejects_absent_arm_before_controller_import(self):
        import importlib.util
        from types import SimpleNamespace
        path=Path('/media/midea/F496AC8096AC4548/act_td3/piper_use_right')
        sys.path.insert(0,str(path))
        spec=importlib.util.spec_from_file_location('right_replay_guard_test',path/'replay_connector.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        for arms,port in [(['left'],'can_arm2'),(['left','right'],'can_arm2'),(['right'],'can_arm1')]:
            with self.assertRaisesRegex(ValueError,'only right arm'):
                module.Hardware(SimpleNamespace(can_right=port),arms).connect(None,None)
        self.assertNotIn('replay_controller',sys.modules)

if __name__=='__main__':unittest.main()

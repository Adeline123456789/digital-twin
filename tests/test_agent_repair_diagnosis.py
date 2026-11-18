import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import agent_loop
import benchmark_collection


def context():
    return dict(source_episode='seed',baseline=dict(offset_mm=0,grasp_tilt_deg=0),
                allowed_grasp_tilt_deg=[-10],prior_failure_warnings=[{'id':'NE011','applicability':'other scene; analogy only'}])


class DiagnosticAgentTests(unittest.TestCase):
    def test_exact_failed_repair_is_scoped_to_source_scene_and_embodiment(self):
        seed=dict(actions=[{'profile':'original'}],active_arms=['right'],initial_state={'task_config_sha256':'scene'},
                  program_version={'source_sha256':{'scripts/try_plug_pick_sim.py':'native'}})
        rejection=dict(source_profile='original',active_arms=['right'],task_config_sha256='scene',native_sha256='native',parameter='grasp_tilt_deg',value=-10)
        self.assertEqual(agent_loop.rejected_tilts(seed,[{'exact_rejection':rejection}]),[-10])
        self.assertEqual(agent_loop.rejected_tilts(seed,[{}, {'exact_rejection':None}]),[])
        for key,value in [('active_arms',['left','right']),('task_config_sha256','other'),('native_sha256','changed'),('source_profile','retained_final')]:
            self.assertEqual(agent_loop.rejected_tilts(seed,[{'exact_rejection':dict(rejection,**{key:value})}]),[])

    def test_only_one_supported_parameter_can_change(self):
        base={'hypotheses':[]}
        for action in [{'grasp_tilt_deg':5},{'grasp_tilt_deg':True},{'grasp_tilt_deg':float('nan')},
                       {'grasp_tilt_deg':-10,'offset_mm':1},{'shell':'execute'}, {'stop':False}]:
            with self.subTest(action=action),self.assertRaises(ValueError):
                agent_loop.validate_diagnostic_plan(dict(base,**action),context())
        self.assertIsNone(agent_loop.validate_diagnostic_plan(dict(base,stop=True),context()))
        self.assertEqual(agent_loop.validate_diagnostic_plan(dict(base,grasp_tilt_deg=-10),context())['offset_mm'],0)

    def test_earlier_retention_loss_blocks_later_ik_probe(self):
        seed=dict(experiment_id='prior',actions=[{'offset_mm':[0,0]}],
            causal_diagnosis={'status':'GRASP_RETENTION_LOSS_OBSERVED'},
            observed_effects=[{'last_phase':'transport','error':'Unreachable pose'}])
        with patch.object(agent_loop,'episode_memory',return_value=[]),patch('probe_transport_ik.probe') as probe:
            result=agent_loop.repair_diagnosis(seed)
        probe.assert_not_called();self.assertEqual(result['allowed_grasp_tilt_deg'],[])
        self.assertIn('Grasp loss precedes',result['next_unknown'])

    def test_actual_agent_entry_passes_diagnosis_and_parameter_to_collector(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for d in ['evolution','protocol','experiments/baseline/20261111T083720Z_f64309a8','experiments/evaluation/seed']:(root/d).mkdir(parents=True,exist_ok=True)
            (root/'protocol/robot_problem_solving.md').write_text('generic')
            old=dict(experiment_id='old',actions=[],observed_effects=[])
            (root/'experiments/baseline/20261111T083720Z_f64309a8/result.json').write_text(json.dumps(old))
            seed=dict(experiment_id='seed',active_arms=['right'],initial_state={'target_index':1,'holder_x_shift_mm':0},actions=[{'offset_mm':[0,0]}],observed_effects=[])
            (root/'experiments/evaluation/seed/result.json').write_text(json.dumps(seed))
            call=dict(answer={'grasp_tilt_deg':-10,'hypotheses':['test, not verified repair']},tool_events=0,usage={'input_tokens':1})
            trial=dict(experiment_id='physics',task_success=False,result='failure',observed_effects=[],execution_time=1,collection_component_seconds=1,recording_verified=True)
            with patch.object(agent_loop,'ROOT',root),patch.object(agent_loop,'record',return_value=(root,{'experiment_id':'candidate'})),patch.object(agent_loop,'finish'),patch.object(agent_loop,'repair_diagnosis',return_value=context()),patch.object(agent_loop,'plan_with_memory',return_value=call) as planner,patch.object(benchmark_collection,'run_case',return_value=trial) as collect,patch('episode_format.source_for'),patch('agent_challenge.precontact_observation',return_value={}),patch('agent_challenge.frame_contract',return_value={}),patch.dict(agent_loop.os.environ,{'PIPER_ACTIVE_ARMS':'right','PIPER_SIM_PROFILE':'original','PIPER_TARGET_INDEX':'1'}):
                result=agent_loop.run('D',0,False,'diagnose_then_repair','seed',max_executions=1)
            self.assertEqual(result['actual_planner_policy'],'diagnose_then_repair')
            planner.assert_called_once();collect.assert_called_once()
            self.assertIn('NE011',planner.call_args.args[0])
            self.assertEqual(collect.call_args.args[0]['grasp_tilt_deg'],-10)
            self.assertEqual(collect.call_args.args[0]['offset_mm'],0)
            self.assertFalse(result['task_success']);self.assertEqual(len(result['diagnostic_contexts']),1)


if __name__=='__main__':unittest.main()

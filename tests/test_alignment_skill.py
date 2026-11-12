import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import alignment_skill as skill
import agent_loop


def context():
    sha=json.loads((skill.ROOT/'evolution/versions/compiled_alignment_v1.json').read_text())['verified_native_sha256']
    return dict(environment='simulation',source_profile='original',previous_source_profile='original',source_target=3,target=3,
        frame='socket',units='mm',align_contact_free=True,retention_status='NO_RETENTION_LOSS_OBSERVED',
        native_sha256=sha,previous_native_sha256=sha,holder_shift_mm=.75,previous_offset_mm=2.,residual=[2.18,-.056])


class SkillTests(unittest.TestCase):
    def test_inapplicable_evidence_is_not_silently_transferred(self):
        base=context()
        for key,value in [('environment','real'),('source_profile','retained_final'),('source_target',1),('target',2),('frame','world'),('units','m'),('align_contact_free',False),('retention_status','GRASP_RETENTION_LOSS_OBSERVED'),('native_sha256','changed'),('holder_shift_mm',2),('residual',[2,.3]),('residual',[float('nan'),0])]:
            with self.subTest(key=key):
                self.assertIsNone(skill.compiled_plan(dict(base,**{key:value}),base['native_sha256']))

    def test_compiled_route_has_no_model_call_and_scope_failure_falls_back(self):
        base=context()
        with patch.object(skill,'ask',return_value={'answer':{},'usage':{},'wall_seconds':1,'tool_events':0}) as ask:
            call=skill.plan_with_memory('same prompt',Path('.'),'compiled_alignment_v1',base)
            ask.assert_not_called();self.assertAlmostEqual(call['answer']['offset_mm'],-.18)
            call=skill.plan_with_memory('same prompt',Path('.'),'compiled_alignment_v1',dict(base,frame='world'))
            self.assertTrue(call['compiled_fallback']);ask.assert_called_once()

    def test_existing_agent_d_reads_active_policy_and_uses_complete_collector(self):
        import benchmark_collection
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for directory in ['evolution','protocol','experiments/baseline/20261111T083720Z_f64309a8','experiments/research_stage3/20261112T075231Z_2a125acc']:(root/directory).mkdir(parents=True,exist_ok=True)
            (root/'evolution/active.json').write_text('{"version":"v2","planner_policy":"compiled_alignment_v1"}')
            (root/'protocol/robot_problem_solving.md').write_text('test protocol')
            seed={'experiment_id':'seed','actions':[],'observed_effects':[]}
            for directory in ['experiments/baseline/20261111T083720Z_f64309a8','experiments/research_stage3/20261112T075231Z_2a125acc']:(root/directory/'result.json').write_text(json.dumps(seed))
            trial=dict(experiment_id='mock_physics',task_success=True,result='success',observed_effects=[],simulation_seconds=1,execution_time=1,collection_component_seconds=1,recording_verified=True)
            with patch.object(agent_loop,'ROOT',root),patch.object(agent_loop,'record',return_value=(root,{'experiment_id':'mock_agent'})),patch.object(agent_loop,'finish'),patch.object(agent_loop,'context_for',return_value=context()),patch.object(benchmark_collection,'run_case',return_value=trial) as collect,patch.object(skill,'ask') as ask,patch.dict(agent_loop.os.environ,{'PIPER_SIM_PROFILE':'original','PIPER_TARGET_INDEX':'3'}):
                result=agent_loop.run('D',.75,publish=False)
            ask.assert_not_called();collect.assert_called_once()
            self.assertTrue(collect.call_args.kwargs['inline']);self.assertFalse(collect.call_args.kwargs['publish'])
            self.assertEqual(result['planner_calls'][0]['planner_kind'],'compiled_alignment_v1')
            self.assertEqual(result['tokens']['input_tokens'],0)


if __name__=='__main__':unittest.main()

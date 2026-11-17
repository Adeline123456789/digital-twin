import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import learn
from bounded_tools import plan


class RuntimeMemoryTests(unittest.TestCase):
    def test_exact_episode_precedes_file_order_and_scope_is_required_for_right(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'memory').mkdir()
            for i in range(7):
                item=dict(id=str(i),signatures=['plug_not_seated'],validation_status='OBSERVED',
                          source_episodes=['source'] if i==6 else [])
                (root/'memory'/f'{i}.json').write_text(json.dumps(item))
            context=dict(episode_id='source',active_arms=['right'],source_profile='original',phase='insert')
            with patch.object(learn,'ROOT',root):
                self.assertEqual([m['id'] for m in learn.retrieve('plug_not_seated',context=context)],['6'])
                item.update(id='scoped',source_episodes=[],retrieval_scope={'active_arms':['right'],'source_profile':'original'})
                (root/'memory'/'scoped.json').write_text(json.dumps(item))
                self.assertEqual([m['id'] for m in learn.retrieve('plug_not_seated',context=context)],['6','scoped'])
                context['source_profile']='retained_final'
                self.assertNotIn('scoped',[m['id'] for m in learn.retrieve('plug_not_seated',context=context)])

    def test_actual_recent_failure_keeps_uncertainty_and_not_transfer_to_dual(self):
        seed=dict(experiment_id='20261117T073201Z_efc6139b',active_arms=['right'],
                  initial_state={'target_index':3},
                  actions=[{'profile':'original'}],observed_effects=[{'last_phase':'fold'}])
        cards=learn.episode_memory(seed)
        card=next(m for m in cards if m['id']=='NE020')
        self.assertIn('Root cause UNKNOWN',card['known_limits'])
        self.assertEqual(card['failure_count'],3)
        self.assertEqual(card['evidence_role'],'HISTORICAL_WARNING_NOT_CURRENT_MEASUREMENT')
        self.assertTrue(card['next_checks'])
        seed['initial_state']['target_index']=2
        self.assertNotIn('NE020',[m['id'] for m in learn.episode_memory(seed)])
        seed['initial_state']['target_index']=3
        seed['active_arms']=['left','right']
        self.assertNotIn('NE020',[m['id'] for m in learn.episode_memory(seed)])

    def test_tool_planner_uses_cards_and_frozen_empty_memory_stays_empty(self):
        class Tools:
            context={'baseline':{'offset_mm':0},'allowed_grasp_tilt_deg':[]}
        seed=dict(experiment_id='fixture',task='insertion',initial_state={},
                  validation_status='UNKNOWN',task_success=None)
        prompts=[]
        def model(prompt,*args,**kwargs):
            prompts.append(json.loads(prompt.split('\n',1)[1]))
            return dict(answer={'plan':{'stop':True,'hypotheses':[]}},usage={},wall_seconds=0,tool_events=0)
        with patch('bounded_tools.episode_memory',return_value=[{'id':'known_failure'}]) as retrieve:
            a=plan(seed,Path('/tmp'),ask_fn=model,dispatcher=Tools(),max_calls=1)
            b=plan(seed,Path('/tmp'),ask_fn=model,dispatcher=Tools(),max_calls=1,memories=[])
        self.assertEqual(retrieve.call_count,1)
        self.assertEqual(prompts[0]['state']['relevant_experience'],[{'id':'known_failure'}])
        self.assertEqual(a['retrieved_experience'],[{'id':'known_failure'}])
        self.assertEqual(prompts[1]['state']['relevant_experience'],[])
        self.assertEqual(b['retrieved_experience'],[])

    def test_default_entry_passes_actual_right_failure_to_planner(self):
        import agent_loop
        from model_compare import ModelCallError
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'experiments/baseline/20261111T083720Z_f64309a8').mkdir(parents=True)
            (root/'experiments/baseline/20261111T083720Z_f64309a8/result.json').write_text('{}')
            (root/'evolution').mkdir();(root/'evolution/active.json').write_text('{"planner_policy":"llm"}')
            (root/'protocol').mkdir();(root/'protocol/robot_problem_solving.md').write_text('fixture')
            prompts=[]
            def capture(prompt,*args,**kwargs):
                prompts.append(prompt)
                raise ModelCallError('Offline boundary: no model or physical execution',{'usage':None})
            with patch.object(agent_loop,'ROOT',root),patch.object(agent_loop,'record',return_value=(root,{'experiment_id':'fixture'})),patch.object(agent_loop,'finish'),patch.object(agent_loop,'plan_with_memory',side_effect=capture),patch.dict(agent_loop.os.environ,{'PIPER_ACTIVE_ARMS':'right','PIPER_SIM_PROFILE':'original','PIPER_TARGET_INDEX':'3'}):
                result=agent_loop.run('D',0,publish=False,max_executions=1)
            self.assertEqual(result['actual_planner_policy'],'llm')
            self.assertIn('NE020',prompts[0])
            self.assertIn('HISTORICAL_WARNING_NOT_CURRENT_MEASUREMENT',prompts[0])
            self.assertIn('NE020',[m['id'] for m in result['retrieved_experience']])
            self.assertIn('Offline boundary',result['error'])


if __name__=='__main__':unittest.main()

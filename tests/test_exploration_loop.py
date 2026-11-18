"""Scripted contract tests; none of these fixtures are model or physics evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import bounded_tools as bt


def episode(ident,offset,success=False):
    return dict(experiment_id=ident,task='fixture',initial_state={},active_arms=['right'],
        actions=[{'offset_mm':[offset,0]}],observed_effects=[{'last_phase':'insert','error':None if success else 'fixture failure'}],
        task_success=success,result='success' if success else 'failure',recording_verified=True,
        validation_status='SIMULATION_VERIFIED' if success else 'SIMULATION_FAILED',
        execution_time=.1,collection_component_seconds=.1)


class FixtureTools:
    def __init__(self,seed,cache,memories):
        self.seed=seed
        self.context=dict(baseline=bt.baseline_case(seed),allowed_grasp_tilt_deg=[])
    def call(self,name,args):
        return dict(value={'episode_id':self.seed['experiment_id'],'actual_fixture_offset':self.context['baseline']['offset_mm']},cache=None)


class ExplorationTests(unittest.TestCase):
    def run_loop(self,answers,results,max_calls=4,max_tools=3):
        inputs=[];executed=[];results=iter(results);answers=iter(answers)
        def ask(prompt,*args,**kwargs):
            inputs.append(json.loads(prompt.split('\n',1)[1]))
            return dict(answer=next(answers),usage={'input_tokens':3,'output_tokens':2},tool_events=0,wall_seconds=.01)
        def execute(case):
            executed.append(case);return next(results)
        with tempfile.TemporaryDirectory() as cache,patch.object(bt,'EpisodeTools',FixtureTools):
            result=bt.explore(episode('seed',0),Path('/tmp'),execute,ask_fn=ask,
                max_calls=max_calls,max_tools=max_tools,cache_directory=cache)
        return result,inputs,executed

    def test_failure_is_next_observation_with_prediction_and_no_budget_reset(self):
        first={'offset_mm':1,'hypotheses':['fixture'],'predicted_effect':'fixture prediction'}
        second={'offset_mm':2,'hypotheses':['revised fixture']}
        result,inputs,executed=self.run_loop([{'plan':first},{'plan':second}],
            [episode('failed_execution',1),episode('successful_execution',2,True)])
        self.assertEqual(inputs[1]['state']['episode_id'],'failed_execution')
        self.assertEqual(inputs[1]['state']['current_parameters']['offset_mm'],1)
        self.assertEqual(inputs[1]['prior_experiments'][0]['proposal'],first)
        self.assertFalse(inputs[1]['prior_experiments'][0]['observed']['task_success'])
        self.assertEqual([p['budget']['calls_remaining'] for p in inputs],[4,3])
        self.assertEqual(result['budget_used'],dict(model_calls=2,tools=0,executions=2))
        self.assertEqual(result['usage']['input_tokens'],6)
        self.assertEqual(result['pending_real_comparisons'],['successful_execution'])
        self.assertEqual([c['offset_mm'] for c in executed],[1,2])

    def test_model_ceiling_is_shared_across_executions(self):
        result,inputs,executed=self.run_loop([{'plan':{'offset_mm':1,'hypotheses':[]}}],
            [episode('failed',1)],max_calls=1)
        self.assertEqual(len(executed),1)
        self.assertEqual(result['status'],'MODEL_BUDGET_EXHAUSTED')

    def test_inspection_budget_is_shared_across_executions(self):
        request={'tool':'trace_window','args':{'center':1,'radius':0},'reason':'fixture'}
        result,inputs,executed=self.run_loop([request,{'plan':{'offset_mm':1,'hypotheses':[]}},request],
            [episode('failed',1)],max_tools=1)
        self.assertEqual(inputs[-1]['budget']['tools_remaining'],0)
        self.assertEqual(result['status'],'TOOL_BUDGET_OR_SCHEMA_REJECTED')
        self.assertEqual(len(executed),1)

    def test_same_failed_experiment_is_not_executed_twice(self):
        request={'plan':{'offset_mm':1,'hypotheses':[]}}
        result,_,executed=self.run_loop([request,request],[episode('failed',1)])
        self.assertEqual(result['status'],'REPEATED_EXPERIMENT_REJECTED')
        self.assertEqual(len(executed),1)

    def test_unverified_recording_stops_and_is_not_a_pending_real_trial(self):
        broken=episode('broken_recording',1,True);broken['recording_verified']=False
        result,_,executed=self.run_loop([{'plan':{'offset_mm':1,'hypotheses':[]}}],[broken])
        self.assertEqual(result['status'],'RECORDING_ERROR')
        self.assertEqual(result['pending_real_comparisons'],[])
        self.assertEqual(len(executed),1)


if __name__=='__main__':unittest.main()

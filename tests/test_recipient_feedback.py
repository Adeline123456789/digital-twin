import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_recipient_feedback as study
from model_compare import ModelCallError


class RecipientStudyTests(unittest.TestCase):
    def test_plan_ready_before_execution_timeout_is_incomplete(self):
        run=dict(label='candidate',planner={'status':'PLAN_READY'},exploration={'status':'TIME_BUDGET_EXHAUSTED'})
        self.assertEqual(study.comparison_status([run])['validation_status'],'COMPARISON_INCOMPLETE')

    def test_teacher_receives_both_recipient_decisions_and_observed_failure(self):
        first=dict(answer={'offset_mm':1},status='PLAN_READY',tool_trace=[],
                   initial_observation={'result':{'value':{'episode_id':'before'}}})
        second=dict(answer={'stop':True},status='STOPPED_UNKNOWN',tool_trace=[])
        feedback=[dict(proposal=first['answer'],observed={'task_success':False})]
        run=dict(planner=second,exploration=dict(planners=[first,second],feedback=feedback),physical_result={'task_success':False})
        value=study.teacher_feedback(run)
        self.assertEqual(len(value['recipient_steps']),2)
        self.assertEqual(value['recipient_steps'][0]['initial_observation']['episode_id'],'before')
        self.assertEqual(value['execution_feedback'],feedback)

    def test_unexecuted_holdout_does_not_count_as_completed_comparison(self):
        runs=[{'label':'heldout parent','planner':{'status':'STOPPED_UNKNOWN'}},
              {'label':'heldout candidate','planner':{'status':'EVIDENCE_OR_REQUEST_ERROR'}}]
        result=study.comparison_status(runs)
        self.assertEqual(result['result'],'infrastructure_error')
        self.assertEqual(result['validation_status'],'COMPARISON_INCOMPLETE')
        self.assertEqual(result['incomplete_arms'],['heldout candidate'])

    def test_resume_keeps_consumed_calls_tokens_simulations_and_time(self):
        with tempfile.TemporaryDirectory() as folder:
            prior={'calls':[{'usage':{'input_tokens':10,'output_tokens':2}}],
                   'simulations':[{'status':'FINISHED'}],'usage_unknown':False}
            b=study.BatchBudget(dict(seconds=5400,model_calls=16,tokens=250000,simulations=8),Path(folder)/'ledger.json',prior,'2026-11-17T00:00:00+00:00')
            self.assertEqual(b.tokens(),12);self.assertEqual(len(b.calls),1);self.assertEqual(len(b.simulations),1)
            self.assertEqual(b.started_at,'2026-11-17T00:00:00+00:00')

    def test_model_token_limit_does_not_reset_or_spend_independent_sim_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            b=study.BatchBudget(dict(seconds=60,model_calls=16,tokens=100,simulations=8),Path(folder)/'ledger.json')
            b.calls=[{'usage':{'input_tokens':101,'output_tokens':0}}]
            with self.assertRaises(RuntimeError):b.check()
            b.check(model=False)
            self.assertEqual(b.tokens(),101);self.assertEqual(b.simulations,[])

    def test_budget_counts_actual_calls_including_failure_and_stops_unknown_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            b=study.BatchBudget(dict(seconds=10,model_calls=2,tokens=100,simulations=1),Path(folder)/'ledger.json')
            call=dict(usage={'input_tokens':3,'output_tokens':2,'cached_input_tokens':3},answer={})
            with patch.object(study,'ask',return_value=call):b.model('x',Path(folder))
            self.assertEqual(b.tokens(),5)
            failed=dict(usage=None,answer=None)
            with patch.object(study,'ask',side_effect=ModelCallError('unavailable',failed)):
                with self.assertRaises(ModelCallError):b.model('x',Path(folder))
            self.assertEqual(len(b.calls),2);self.assertTrue(b.usage_unknown)
            with self.assertRaises(RuntimeError):b.check()
            self.assertEqual(len(json.loads(b.path.read_text())['calls']),2)

    def test_stopping_is_not_counted_as_physical_success(self):
        call={'answer':{'stop':True,'hypotheses':[]},'applicability':{},'status':'STOPPED_UNKNOWN','tool_trace':[]}
        budget=Mock()
        loop=dict(planners=[call],experiments=[],feedback=[],status='STOPPED_UNKNOWN')
        with patch.object(study,'explore',return_value=loop) as explore:
            result=study.execute_student({'experiment_id':'seed'},'guidance','test',0,budget,Path('/tmp'),'student')
        self.assertIsNone(result['physical_result'])
        budget.simulate.assert_not_called()
        self.assertIs(explore.call_args.kwargs['ask_fn'],budget.model)
        self.assertEqual(explore.call_args.kwargs['max_executions'],2)

    def test_teacher_context_contains_recipient_feedback_without_heldout_or_private_state(self):
        record=dict(planner={'answer':{'stop':True},'status':'STOPPED_UNKNOWN','tool_trace':[]},
                    physical_result={'task_success':False},native_outcome={'error':'retention loss'},
                    heldout={'answer':'never expose'},private_camera={'secret':'never expose'})
        feedback=study.teacher_feedback(record)
        self.assertFalse(feedback['physical_result']['task_success'])
        self.assertNotIn('heldout',json.dumps(feedback));self.assertNotIn('private_camera',feedback)

    def test_runner_orders_recipient_then_teacher_and_keeps_teacher_out_of_holdout(self):
        order=[]
        class Budget:
            calls=[];simulations=[];usage_unknown=False
            def __init__(self,*args):pass
            def write(self):pass
            def tokens(self):return 0
            def simulate(self,case,label):
                order.append(label);return {'experiment_id':label}
            def model(self,*args,**kwargs):
                order.append('teacher');return {'tool_events':0,'answer':{'guidance':'revised','change_reason':'feedback','direct_solution':False}}
        def student(seed,guidance,label,*args):
            order.append(label)
            self.assertEqual(args[-1],1)
            return {'label':label,'planner':{'answer':{'stop':True},'status':'STOPPED_UNKNOWN','tool_trace':[]},'physical_result':None}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'protocol').mkdir();(root/'protocol/robot_problem_solving.md').write_text('generic')
            out=root/'report';out.mkdir()
            source=root/'source.json';source.write_text(json.dumps({'active_arms':['right'],'initial_state':{'target_index':3},'actions':[{'offset_mm':[0,0]}],'observed_effects':[{}]}))
            design=root/'design.json';design.write_text(json.dumps({'hardware_executions':0,'recipient_executions':1,'source_episode':'source','task_sha256':'fixed','limits':{},'student':'student','teacher':'teacher','development':{'parameters':{},'layout_shift_mm':0},'heldout':{'parameters':{},'layout_shift_mm':1}}))
            with patch.object(study,'ROOT',root),patch.object(study,'digest',return_value='fixed'),patch('episode_evidence.locate',return_value=(source,root,root)),patch('episode_format.source_for'),patch.object(study,'record',return_value=(out,{'experiment_id':'test'})),patch.object(study,'finish'),patch.object(study,'BatchBudget',Budget),patch.object(study,'execute_student',side_effect=student),patch.dict(study.os.environ,{}):
                result=study.run(design)
            self.assertEqual(result['result'],'success')
            self.assertEqual(order,['development seed','recipient before teacher','teacher','recipient after feedback','heldout seed','heldout parent','heldout candidate'])
            self.assertEqual(result['promotion_status'],'NOT_PROMOTED')


if __name__=='__main__':unittest.main()

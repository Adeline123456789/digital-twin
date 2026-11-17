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
        with patch.object(study,'plan',return_value=call):
            result=study.execute_student({'experiment_id':'seed'},'guidance','test',0,budget,Path('/tmp'),'student')
        self.assertIsNone(result['physical_result'])
        budget.simulate.assert_not_called()

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
            return {'label':label,'planner':{'answer':{'stop':True},'status':'STOPPED_UNKNOWN','tool_trace':[]},'physical_result':None}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'protocol').mkdir();(root/'protocol/robot_problem_solving.md').write_text('generic')
            out=root/'report';out.mkdir()
            source=root/'source.json';source.write_text(json.dumps({'active_arms':['right'],'initial_state':{'target_index':3},'actions':[{'offset_mm':[0,0]}],'observed_effects':[{}]}))
            design=root/'design.json';design.write_text(json.dumps({'hardware_executions':0,'source_episode':'source','task_sha256':'fixed','limits':{},'student':'student','teacher':'teacher','development':{'parameters':{},'layout_shift_mm':0},'heldout':{'parameters':{},'layout_shift_mm':1}}))
            with patch.object(study,'ROOT',root),patch.object(study,'digest',return_value='fixed'),patch('episode_evidence.locate',return_value=(source,root,root)),patch('episode_format.source_for'),patch.object(study,'record',return_value=(out,{'experiment_id':'test'})),patch.object(study,'finish'),patch.object(study,'BatchBudget',Budget),patch.object(study,'execute_student',side_effect=student),patch.dict(study.os.environ,{}):
                result=study.run(design)
            self.assertEqual(result['result'],'success')
            self.assertEqual(order,['development seed','recipient before teacher','teacher','recipient after feedback','heldout seed','heldout parent','heldout candidate'])
            self.assertEqual(result['promotion_status'],'NOT_PROMOTED')


if __name__=='__main__':unittest.main()

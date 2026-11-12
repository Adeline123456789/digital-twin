import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_transport_repair as runner


class BudgetTests(unittest.TestCase):
    def run_with_candidate(self, candidate_success):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'experiments/research_stage4/round2';out.mkdir(parents=True)
            (out/'design.json').write_text(json.dumps({'full_execution':{'cases':[
                {'shift_mm':0,'order':['baseline','candidate'],'role':'development'},
                {'shift_mm':.5,'order':['candidate','baseline'],'role':'heldout'}]}}))
            (out/'selection.json').write_text('{"selected_grasp_tilt_deg":-10}')
            seen=[]
            def fake_run(case,**kwargs):
                seen.append(case)
                return {'experiment_id':str(len(seen)),'task_success':candidate_success if case['arm']=='candidate' else False,'recording_verified':True}
            with patch.object(runner,'ROOT',root),patch.object(runner,'run_case',side_effect=fake_run),patch.dict(runner.os.environ,{'PIPER_SIM_PROFILE':'retained_final','PIPER_TARGET_INDEX':'1'}):
                runner.main()
            return seen,json.loads((out/'execution_progress.json').read_text())

    def test_failed_candidate_stops_before_holdout(self):
        seen,status=self.run_with_candidate(False)
        self.assertEqual(len(seen),2)
        self.assertEqual(status['status'],'STOPPED_CANDIDATE_NOT_VALIDATED')

    def test_success_keeps_matched_budget_and_single_parameter(self):
        seen,status=self.run_with_candidate(True)
        self.assertEqual(status['status'],'COMPLETED')
        self.assertEqual([x['grasp_tilt_deg'] for x in seen],[0,-10,-10,0])
        self.assertEqual([x['shift_mm'] for x in seen],[0,0,.5,.5])
        self.assertTrue(all(x['offset_mm']==0 for x in seen))


if __name__=='__main__':unittest.main()

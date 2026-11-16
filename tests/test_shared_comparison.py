import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_shared_reference import compare
from trajectory_contract import RIGHT,sha

class SharedComparisonTests(unittest.TestCase):
    def test_measures_timing_drift_and_rejects_modified_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=root/'reference';real=root/'real';ref.mkdir();real.mkdir()
            targets=np.zeros((6,7));targets[:,0]=np.arange(6)*.1;targets[:,6]=.5
            np.savez(ref/'reference.npz',time=np.arange(6)*.01,targets=targets,phases=['approach']*6)
            ident=sha(ref/'reference.npz')
            (ref/'manifest.json').write_text(json.dumps(dict(trajectory_id=ident,feature_names=RIGHT)))
            prediction=[dict(time=i*.01,q=[np.deg2rad(i*.1)]+[0]*5+[.0475]) for i in range(6)]
            (ref/'prediction.jsonl').write_text('\n'.join(map(json.dumps,prediction)))
            (ref/'validation.json').write_text(json.dumps(dict(reference_sha256=ident,task_success=True,all_substeps_valid=True,prediction_sha256=sha(ref/'prediction.jsonl'))))
            (real/'execution_report.json').write_text(json.dumps(dict(reference_id=ident,physical_task_success=None)))
            path=real/'commanded_probe.csv'
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['monotonic','phase']+[f'{kind}_j{i}' for kind in ['target','feedback'] for i in range(1,7)]+['target_gripper_open','feedback_gripper_open'])
                for i,q in enumerate(targets):w.writerow([100+i*.02,'approach',*q[:6],*(q[:6]+3),q[6],q[6]])
            result=compare(ref,real)
            self.assertTrue(result['entire_reference_sent'])
            self.assertAlmostEqual(result['final_logged_time_lag_s'],.05)
            self.assertEqual(result['first_persistent_tracking_deviation']['sample'],0)
            self.assertIsNone(result['physical_task_success'])
            self.assertEqual(result['root_cause'],'UNKNOWN')
            with path.open() as f:rows=list(csv.reader(f))
            rows[1][2]='.01'
            with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
            with self.assertRaises(ValueError):compare(ref,real)

if __name__=='__main__':unittest.main()

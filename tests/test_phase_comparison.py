import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_replays import phase_comparison,contact_summary


class TraceTests(unittest.TestCase):
    def test_onset_and_abort_are_separate(self):
        phases=['grasp']*5+['lift']*5+['abort_retreat']*5
        rows=[{'phase':p,'plug_in_gripper_m':[0,0,0],'metrics':{'success':False}} for p in phases]
        error=np.r_[np.zeros(5),np.ones(5)*4,np.ones(5)*20]
        trace=phase_comparison(rows,np.arange(15)/30,np.zeros((15,6)),error)
        self.assertEqual(trace['gripper_divergence_onset'][0]['first']['phase'],'lift')
        self.assertIsNone(trace['gripper_divergence_onset'][2]['first'])
        self.assertAlmostEqual(trace['phase_trace'][2]['gripper_rmse_mm'],20)

    def test_proximity_is_not_active_contact(self):
        row={'phase':'lift','objects':{'yellow_plug':{'position':[0,0,1]}},'task_metrics':{},
             'contacts':[{'body1':'task/yellow_plug','body2':'finger','distance_m':.001}]}
        self.assertEqual(contact_summary([row],30)[0]['frames_with_plug_contact'],0)
        row['contacts'][0]['distance_m']=-.00001
        self.assertEqual(contact_summary([row],30)[0]['contact_body_frame_counts'],{'finger':1})

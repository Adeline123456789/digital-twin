import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_retention import analyze_retention


class RetentionTests(unittest.TestCase):
    def rows(self,phase='transport'):
        return [dict(phase='lift',time=0,plug=[0,0,0],grip=[0,0,.01])]+[
            dict(phase=phase,time=i/30,plug=[0,0,0],grip=[0,0,.05]) for i in [1,2,3]]

    def test_loss_needs_sustained_separation_and_contact_loss(self):
        rows=self.rows();contacts=[{'contacts':[]} for _ in rows]
        self.assertEqual(analyze_retention(rows,contacts)['earliest_observed_frame'],1)
        contacts[2]['contacts']=[{'body1':'task/yellow_plug','body2':'right/link7'}]
        self.assertEqual(analyze_retention(rows,contacts)['status'],'NO_RETENTION_LOSS_OBSERVED')

    def test_intentional_release_is_not_carry_loss(self):
        rows=self.rows('release')
        self.assertEqual(analyze_retention(rows,[{'contacts':[]} for _ in rows])['status'],'UNKNOWN')

    def test_missing_or_misaligned_evidence_not_success(self):
        self.assertEqual(analyze_retention([],[])['status'],'UNKNOWN')
        with self.assertRaises(AssertionError):analyze_retention(self.rows(),[])

if __name__=='__main__':unittest.main()

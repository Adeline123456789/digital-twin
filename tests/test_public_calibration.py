"""The observed private-calibration publishing regression, without robot access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from publish import check_private_calibration
from fit_probe_registration import public_summary


class PublicCalibrationTests(unittest.TestCase):
    def test_reject_nested_device_transforms(self):
        for field in ('T_link6_rgb','baseline_T_link6_rgb','T_parent_rgb'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                check_private_calibration({'results':[{'camera':{field:[[1,0],[0,1]]}}]})

    def test_public_copy_preserves_evidence_and_local_original(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fit_001/result.json';p.parent.mkdir()
            private={'result':'failure','calibration_promoted':False,'candidate':{'median_px':2.1},
                     'T_link6_rgb':[[1,0],[0,1]],'baseline_T_link6_rgb':[[1,0],[0,1]]}
            p.write_text(json.dumps(private));before=p.read_bytes()
            public=public_summary(p);check_private_calibration(public)
            self.assertEqual(public['candidate'],private['candidate'])
            self.assertEqual(public['result'],'failure')
            self.assertEqual(len(public['private_calibration_evidence']['sha256']),64)
            self.assertEqual(p.read_bytes(),before)


if __name__=='__main__':unittest.main()

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evidence_cache import EvidenceCache
from real_episode_evidence import summarize_capture


class RealEvidenceTests(unittest.TestCase):
    def test_cache_preserves_unknown_and_invalidates_changed_review_and_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'STOP').touch()
            (root / 'execution_report.json').write_text(json.dumps(dict(
                physical_task_success=None, completed_sequence=True)))
            fields = ['monotonic', 'phase', 'target_gripper_open', 'feedback_gripper_open']
            with (root / 'commanded_probe.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                for i in range(6):
                    writer.writerow(dict(monotonic=10+i/30, phase='lift' if i < 3 else 'insert',
                        target_gripper_open=.3, feedback_gripper_open=.4))
            samples = [dict(rgb_device_ms=(1000+i/30)*1000, host_monotonic=10+i/30+.02,
                host_unix=1000+i/30+.02, rgb_frame=i, timestamp_domain='timestamp_domain.global_time')
                for i in range(6)]
            for camera in ('external', 'right_wrist'):
                (root / camera).mkdir()
                (root / camera / 'camera.json').write_text(json.dumps(dict(samples=samples)))
            cache = EvidenceCache(root / 'cache')
            try:
                cold = summarize_capture(root, cache)
                warm = summarize_capture(root, cache)
                self.assertEqual(cold['value'], warm['value'])
                self.assertTrue(warm['cache']['hit'])
                self.assertIsNone(warm['value']['physical_task_success'])
                self.assertEqual(warm['value']['phases'][0]['end_views']['right_wrist']['path'], 'right_wrist/0002.jpg')
                (root / 'observed_execution_report.json').write_text(json.dumps(dict(physical_task_success=False)))
                reviewed = summarize_capture(root, cache)
                self.assertFalse(reviewed['cache']['hit'])
                self.assertIs(reviewed['value']['physical_task_success'], False)
                samples[4]['rgb_device_ms'] += 250
                (root / 'right_wrist/camera.json').write_text(json.dumps(dict(samples=samples)))
                rejected = summarize_capture(root, cache)
                self.assertFalse(rejected['cache']['hit'])
                self.assertNotIn('right_wrist', rejected['value']['phases'][0]['end_views'])
                self.assertEqual(rejected['value']['camera_clocks']['right_wrist']['first_outside_one_frame'], 4)
            finally:
                cache.close()

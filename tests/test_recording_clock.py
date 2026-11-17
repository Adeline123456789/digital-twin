import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recording_clock import map_camera_clock


class RecordingClockTests(unittest.TestCase):
    def samples(self, jump):
        return [dict(rgb_device_ms=(1000+i/30)*1000,host_monotonic=10+i/30+.02,
                     host_unix=1000+i/30+.02+(jump if i>=3 else 0)) for i in range(6)]

    def test_wallclock_step_does_not_shift_camera_to_future_commands(self):
        normal, _ = map_camera_clock(self.samples(0))
        stepped, report = map_camera_clock(self.samples(-.813))
        np.testing.assert_allclose(normal, stepped)
        np.testing.assert_allclose(stepped, 10+np.arange(6)/30)
        self.assertAlmostEqual(report['max_change_from_per_frame_wall_mapping_s'],.813)

    def test_unstable_device_time_is_not_silently_repaired(self):
        samples=self.samples(0)
        for sample in samples[3:]:sample['rgb_device_ms']+=800
        with self.assertRaisesRegex(ValueError,'not continuous'):map_camera_clock(samples)

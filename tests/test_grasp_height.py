import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import learn


class GraspHeightTests(unittest.TestCase):
    def test_collector_receives_and_records_candidate_height(self):
        for height in (None, .012, float('nan'), .001):
            with self.subTest(height=height), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = {'experiment_id': 'case'}
                with patch.object(learn, 'ROOT', root), patch.object(learn, 'DATA', root / 'data'), \
                     patch.object(learn, 'record', return_value=(root, result)), \
                     patch.object(learn, 'finish'), patch.dict(learn.os.environ, {'PIPER_SIM_PROFILE':'original', 'PIPER_ACTIVE_ARMS':'right'}), \
                     patch.object(learn.subprocess, 'run', side_effect=RuntimeError('inspected command')) as run:
                    learn.simulate('evaluation', 0, 0, 'height', record_inline=True, grasp_height_m=height)
                if height is None or height == .012:
                    command = run.call_args.args[0]
                    self.assertEqual(float(command[command.index('--grasp-height') + 1]), height or .020)
                    self.assertEqual(result['actions'][0]['grasp_height_m'], height or .020)
                else:
                    run.assert_not_called()


if __name__ == '__main__':
    unittest.main()

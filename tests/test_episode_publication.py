import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import archive_trial
import episode_format

class PublicationTests(unittest.TestCase):
    def test_failed_conversion_never_publishes_source_only_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'data';ident='test_failure'
            report=root/'experiments/evaluation'/ident/'result.json';report.parent.mkdir(parents=True)
            report.write_text(json.dumps({'environment':'simulation','task_success':False}))
            (data/'self_evolution'/ident).mkdir(parents=True)
            (data/'self_evolution'/ident/'trajectory.jsonl').write_text('{}\n')
            with patch.object(archive_trial,'ROOT',root),patch.object(archive_trial,'DATA',data),patch.object(sys,'argv',['archive_trial.py',ident]),patch.object(episode_format,'source_for',return_value=root),patch.object(episode_format,'convert_group',side_effect=RuntimeError('render failed')):
                with self.assertRaisesRegex(RuntimeError,'render failed'):archive_trial.main()
            self.assertFalse((data/'complete_sim/failed'/ident).exists())

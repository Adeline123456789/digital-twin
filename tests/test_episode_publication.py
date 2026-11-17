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
    def test_real_exporter_change_does_not_invalidate_unchanged_simulation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'scripts').mkdir()
            sim=root/'scripts/try_plug_pick_sim.py';sim.write_text('unchanged simulation')
            export=root/'scripts/export_plug_calibration.py';export.write_text('old exporter')
            record={'program_version':{'source_sha256':{str(p.relative_to(root)):episode_format.digest(p) for p in (sim,export)}}}
            export.write_text('fixed real clock mapping')
            with patch.object(episode_format,'ORIGINAL',root):
                self.assertEqual(episode_format.source_for(record),root)
                sim.write_text('different physics program')
                with self.assertRaisesRegex(ValueError,'try_plug_pick_sim.py'):episode_format.source_for(record)

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

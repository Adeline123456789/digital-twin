import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from organize_success_episodes import organize, LEGACY
import archive_trial
import finalize_recordings
from contextlib import nullcontext
from types import SimpleNamespace


class SuccessLayoutTests(unittest.TestCase):
    def test_migration_keeps_bytes_groups_episode_and_separates_source_only(self):
        with tempfile.TemporaryDirectory() as td:
            data=Path(td)/'data';base=data/'complete_sim/succeed'/LEGACY
            contents={
                'videos/original.mp4':b'original-video',
                'proprio_stats/original.h5':b'original-state',
                'videos/recorded/cam.mp4':b'recorded-video',
                'proprio_stats/recorded/replay/proprio_stats.h5':b'recorded-state',
                'proprio_stats/recorded/replay/outcome.json':b'{"success":true}',
                'proprio_stats/recorded/source_trial/trajectory.jsonl':b'original-trace',
                'proprio_stats/intermediate/source_trial/trace':b'headless-trace',
                'recording_recorded.json':b'recording-report'}
            for name,body in contents.items():
                p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
            failed=data/'complete_sim/failed/untouched';failed.mkdir(parents=True)
            (failed/'outcome.json').write_bytes(b'failed')
            report=organize(data,Path(td)/'migration.json',apply=True)
            self.assertEqual(report['recorded_successes_moved'],['recorded'])
            for row in report['files']:
                old=Path(row['old']).relative_to(base.relative_to(data))
                self.assertEqual((data/row['new']).read_bytes(),contents[str(old)])
            episode=data/'complete_sim/succeed/recorded'
            self.assertTrue((episode/'videos/cam.mp4').is_file())
            self.assertTrue((episode/'proprio_stats/proprio_stats.h5').is_file())
            self.assertTrue((episode/'outcome.json').is_file())
            self.assertFalse((base/'videos/recorded').exists())
            self.assertTrue((data/'self_evolution/legacy_success_evidence/intermediate/source_trial/trace').is_file())
            self.assertEqual((failed/'outcome.json').read_bytes(),b'failed')

    def test_new_recording_places_all_modalities_under_one_episode(self):
        with tempfile.TemporaryDirectory() as td:
            data=Path(td);ident='new_success'
            staged=data/'stage/complete_sim/succeed'/('learning_'+ident+'_000001')
            (staged/'videos').mkdir(parents=True);(staged/'proprio_stats').mkdir()
            (staged/'outcome.json').write_text('{"success":true}')
            (staged/'proprio_stats/metadata.json').write_text('{"num_frames":1}')
            (staged/'proprio_stats/proprio_stats.h5').write_bytes(b'states')
            for camera in ('external','left_wrist','right_wrist'):
                (staged/'videos'/('cam_'+camera+'_rgb.mp4')).write_bytes(b'video')
            record={}
            video=SimpleNamespace(decode=lambda **kw: iter([object()]))
            with patch.object(finalize_recordings,'DATA',data), patch.object(finalize_recordings,'SUCCESS',data/'complete_sim/succeed'), patch.object(finalize_recordings,'record',return_value=(data/'report',record)), patch.object(finalize_recordings,'finish'), patch.object(finalize_recordings,'supplement'), patch.object(finalize_recordings,'validate',return_value={'status':'verified fixture'}), patch('av.open',side_effect=lambda *a,**kw:nullcontext(video)), patch.object(sys,'argv',['finalize_recordings.py',ident,'--stage','stage']):
                finalize_recordings.main()
            episode=data/'complete_sim/succeed'/ident
            self.assertEqual(record['result'],'success')
            self.assertTrue((episode/'videos/cam_external_rgb.mp4').is_file())
            self.assertTrue((episode/'proprio_stats/proprio_stats.h5').is_file())
            self.assertTrue((episode/'outcome.json').is_file())
            self.assertFalse((data/'complete_sim/succeed'/LEGACY).exists())

    def test_headless_success_is_archived_outside_formal_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'data';ident='headless_success'
            report=root/'experiments/evaluation'/ident/'result.json';report.parent.mkdir(parents=True)
            report.write_text(json.dumps({'environment':'simulation','task_success':True}))
            raw=data/'self_evolution'/ident;raw.mkdir(parents=True)
            (raw/'trajectory.jsonl').write_text('{}\n')
            with patch.object(archive_trial,'ROOT',root),patch.object(archive_trial,'DATA',data),patch.object(sys,'argv',['archive_trial.py',ident]):
                archive_trial.main()
            self.assertFalse((data/'complete_sim/succeed').exists())
            self.assertTrue((data/'self_evolution/archived_trials'/ident/'source_trial/trajectory.jsonl').is_file())


if __name__=='__main__':unittest.main()

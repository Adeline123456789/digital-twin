"""Co-locate legacy success recordings without changing any existing file bytes."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from learn import DATA, digest, save

LEGACY = '20261111_sim_calibrated_pick_insert_000001'


def organize(data, report_path, apply=False):
    root = data / 'complete_sim/succeed'
    legacy = root / LEGACY
    video_ids = {p.name for p in (legacy/'videos').iterdir() if p.is_dir()}
    stats_ids = {p.name for p in (legacy/'proprio_stats').iterdir() if p.is_dir()}
    assert video_ids <= stats_ids, 'Recording without matching state directory'
    for ident in video_ids:
        assert not (root/ident).exists(), 'Destination already exists: '+ident
        assert (legacy/'proprio_stats'/ident/'replay/proprio_stats.h5').is_file()
    moves = []
    for old in sorted(legacy.rglob('*')):
        if not old.is_file():
            continue
        rel = old.relative_to(legacy)
        parts = rel.parts
        new = old
        if parts[0] == 'videos' and len(parts) > 2:
            new = root/parts[1]/'videos'/Path(*parts[2:])
        elif parts[0] == 'proprio_stats' and len(parts) > 2:
            ident = parts[1]
            if ident not in video_ids:
                new = data/'self_evolution/legacy_success_evidence'/ident/Path(*parts[2:])
            elif parts[2] == 'replay':
                new = root/ident/('outcome.json' if parts[3] == 'outcome.json' else 'proprio_stats/'+str(Path(*parts[3:])))
            elif parts[2] == 'outcome.json':
                new = root/ident/'source_evidence_outcome.json'
            else:
                new = root/ident/Path(*parts[2:])
        elif parts[0].startswith('recording_'):
            ident = parts[0][len('recording_'):-len('.json')]
            assert ident in video_ids
            new = root/ident/parts[0]
        if new != old:
            assert not new.exists(), str(new)
        moves.append({'old':str(old.relative_to(data)), 'new':str(new.relative_to(data)),
                      'sha256':digest(old), 'bytes':old.stat().st_size})
    assert len({row['new'] for row in moves}) == len(moves)
    report = {'timestamp':datetime.now(timezone.utc).isoformat(),
              'operation':'Storage-only relocation; existing content unchanged',
              'recorded_successes_moved':sorted(video_ids),
              'source_only_trials_relocated':sorted(stats_ids-video_ids),
              'original_episode_preserved':LEGACY, 'files':moves, 'applied':False}
    assert not report_path.exists(), 'Use a unique migration report'
    save(report_path,report)  # Persist the complete plan before the first rename.
    if not apply:
        return report
    for row in moves:
        old, new = data/row['old'], data/row['new']
        if old != new:
            new.parent.mkdir(parents=True,exist_ok=True)
            old.rename(new)
        assert digest(new) == row['sha256'], 'Content changed: '+row['new']
    generated=[]
    for ident in sorted(video_ids):
        episode=root/ident
        if not (episode/'outcome.json').exists():
            # Older recordings only had the headless archive index. Derive the
            # summary from their existing recorded metadata, preserving the index.
            meta=json.loads((episode/'proprio_stats/metadata.json').read_text())
            assert meta['success'] is True
            save(episode/'outcome.json',dict(domain='sim',success=True,
                source_session=meta['session_id'],source_episode=meta['episode_id'],
                provenance='Derived from existing recording metadata during storage migration'))
            generated.append(str((episode/'outcome.json').relative_to(data)))
    for directory in sorted((p for p in legacy.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    report.update(applied=True,all_existing_file_hashes_preserved=True,
                  generated_outcome_summaries=generated)
    save(report_path,report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    result=organize(DATA,args.report,args.apply)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))

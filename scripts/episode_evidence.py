"""Phase summaries and requested windows, using existing recordings without video decoding."""
import json
import math
import re
from itertools import islice
from pathlib import Path
from learn import ROOT, DATA
from diagnose_retention import analyze_retention


def locate(ident):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', ident):
        raise ValueError('Expected an Episode ID, not a file path')
    reports = list((ROOT / 'experiments').glob('*/' + ident + '/result.json'))
    episodes = [DATA / 'complete_sim' / outcome / ident for outcome in ('succeed', 'failed')]
    episodes = [p for p in episodes if p.is_dir()]
    if len(reports) != 1 or len(episodes) != 1:
        raise ValueError('Tool needs one archived simulation Episode and its source result')
    return reports[0], episodes[0], DATA / 'self_evolution' / ident


def json_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def frame_at(path, frame):
    if type(frame) is not int or frame < 0:
        raise ValueError('Frame must be a nonnegative integer')
    with path.open() as stream:
        row = next(islice(stream, frame, frame + 1), None)
    if row is None:
        raise ValueError('Frame outside recording')
    return json.loads(row)


def compact_frame(row, extra, index):
    pairs = sorted({tuple(sorted((c['body1'], c['body2']))) for c in extra['contacts']
                    if 'task/yellow_plug' in (c['body1'], c['body2'])})
    return dict(frame=index, phase=row['phase'], time_s=row['time'],
                plug_gripper_separation_mm=math.dist(row['plug'], row['grip'])*1000,
                plug_position_world_m=row['plug'], gripper_position_world_m=row['grip'],
                recorded_contact_pairs=[list(pair) for pair in pairs], contact_force_status='NOT_RECORDED',
                task_metrics=extra.get('task_metrics', {}))


def summarize(result, rows, privileged):
    if not rows or len(rows) != len(privileged):
        raise ValueError('Trace and privileged samples must align')
    endpoints = [i for i, row in enumerate(rows)
                 if i == len(rows)-1 or rows[i+1]['phase'] != row['phase']]
    native = result['observed_effects'][0]
    return dict(episode_id=result['experiment_id'], scope='Simulation-only',
        task_success=result['task_success'], recording_verified=result.get('recording_verified'),
        validation_status=result['validation_status'], active_arms=result.get('active_arms'),
        source_versions=dict(scene=result['initial_state'].get('task_config_sha256'),
            native_program=result['program_version'].get('source_sha256', {}).get('scripts/try_plug_pick_sim.py'),
            camera='Not used by this state/contact analyzer', controller='Recorded simulation actions'),
        actions=result['actions'], native_error=native.get('error'), last_phase=native.get('last_phase'),
        phases=[compact_frame(rows[i], privileged[i], i) for i in endpoints],
        retention=analyze_retention(rows, privileged), root_cause='UNKNOWN',
        unknowns=['Real contact force and motor-torque mapping', 'Historical applied forces and direct state edits',
                  'Recorded contact pair alone does not prove active constraint or gripping force'],
        privileged_information_used=True)


def summary(ident, cache):
    report, episode, raw = locate(ident)
    deps = dict(report=report, trajectory=raw/'trajectory.jsonl',
                privileged=episode/'proprio_stats/privileged.jsonl', analyzer=Path(__file__),
                retention=ROOT/'scripts/diagnose_retention.py')
    return cache.get('episode_summary_v1', deps, {}, lambda: summarize(
        json.loads(report.read_text()), json_rows(deps['trajectory']), json_rows(deps['privileged'])))


def window(ident, center, radius, cache):
    if type(center) is not int or center < 0 or type(radius) is not int or not 0 <= radius <= 15:
        raise ValueError('Use a nonnegative frame and radius 0..15')
    _, episode, raw = locate(ident)
    deps = dict(trajectory=raw/'trajectory.jsonl', privileged=episode/'proprio_stats/privileged.jsonl', analyzer=Path(__file__))
    def compute():
        output = []
        with deps['trajectory'].open() as a, deps['privileged'].open() as b:
            for i, (line, extra) in enumerate(zip(a, b, strict=True)):
                if i < max(0, center-radius): continue
                if i > center+radius: break
                output.append(compact_frame(json.loads(line), json.loads(extra), i))
        if not any(r['frame'] == center for r in output): raise ValueError('Frame outside recording')
        return dict(episode_id=ident, frames=output, privileged_information_used=True)
    return cache.get('episode_window_v1', deps, dict(center=center, radius=radius), compute)

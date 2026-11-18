"""Same-file paired diagnostics. Write metrics to stdout, no dataset mutations."""
import json
import math
import sys
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[2]
sys.path.insert(0,str(root/'scripts'))
from triage_episode import replay
from learn import digest


def wilson(k,n):
    z=1.959963984540054
    mid=(k/n+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [0. if k==0 else max(0.,mid-half),1. if k==n else min(1.,mid+half)]


def measure(fresh):
    base=json.loads((out.parent/'B1/measurements.json').read_text())
    paths=[(root/c['report'],c['split']) for c in base['cases']]
    for slot in fresh:
        if slot['recording_verified']:
            paths.append((root/'experiments/fast_slow_collection'/slot['experiment_id']/'result.json','new_collection'))
    rows=[]
    for path,split in paths:
        r=json.loads(path.read_text())
        a=replay(path,legacy=True)
        b=replay(path)
        status=a['diagnosis']['status']
        expected={'NO_RETENTION_LOSS_OBSERVED':'NO_DIVERGENCE_OBSERVED'}.get(status,status)
        rows.append(dict(episode=r['experiment_id'],split=split,profile=r['actions'][0]['profile'],
                         native=r['program_version']['source_sha256']['scripts/try_plug_pick_sim.py'],target=r['initial_state']['target_index'],
                         report_sha256=digest(path),trajectory_sha256=next(x['sha256'] for x in r['evidence'] if x['dataset_relative_path'].endswith('/trajectory.jsonl')),
                         native_success=r['task_success'],legacy_status=status,fastslow_status=b['diagnosis']['status'],
                         equal_native_label=b['diagnosis']['task_success'] is a['diagnosis']['task_success_unchanged'],
                         equivalent_diagnosis=expected==b['diagnosis']['status'],decision=b['decision']['decision'],
                         frames=b['measured_frames'],legacy_wall_s=a['diagnostic_wall_seconds'],fastslow_wall_s=b['diagnostic_wall_seconds'],
                         latency_ms=b['processing_ms'],confirmation_delay_s=b['detection_delay_s'],
                         input_tokens=b['input_tokens'],output_tokens=b['output_tokens'],planner_calls=int(b['model_invoked'])))
    controls=[r for r in rows if r['split']=='heldout_success_episodes']
    alarms=sum(r['fastslow_status']=='GRASP_RETENTION_LOSS_OBSERVED' for r in controls)
    unknown=sum(r['fastslow_status']=='UNKNOWN' for r in controls)
    return dict(rows=rows,new_collections=fresh,heldout_controls=dict(n=len(controls),false_alarms=alarms,
                unique_trajectories=len({r['trajectory_sha256'] for r in controls}),unknown=unknown,
                population_interval=None,reason='Repeated deterministic traces and nonrandom sampling preclude population inference',
                illustrative_wilson95_if_independent=wilson(alarms,len(controls))),
                unique_trajectories=len({r['trajectory_sha256'] for r in rows}),independent_positive_recall='NOT_MEASURED',fake_success_rate='NOT_MEASURED',
                native_detector_changed=False,real_status='REAL_NOT_TESTED',real_root_cause='UNKNOWN',
                max_per_episode_p95_ms=max(r['latency_ms']['p95'] for r in rows if r['frames']),
                full_trace_units='per-frame step after file loading and alignment; full wall includes validation and reading',
                all_native_labels_equal=all(r['equal_native_label'] for r in rows),
                known_trace_parity=all(r['equivalent_diagnosis'] for r in rows))


if __name__=='__main__':
    slots=[json.loads(p.read_text()) for p in sorted(out.glob('*_result.json'))]
    print(json.dumps(measure(slots),indent=2))

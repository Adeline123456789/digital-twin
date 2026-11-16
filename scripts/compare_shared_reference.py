"""Compare a frozen-reference prefix with real sent commands and feedback; never fit or warp time."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from trajectory_contract import RIGHT,load_reference,sha


def compare(reference,capture):
    reference,capture=Path(reference),Path(capture)
    manifest=json.loads((reference/'manifest.json').read_text())
    if manifest['feature_names']!=RIGHT:
        raise ValueError('This validator contract is the canonical right-only joint/width layout')
    execution=json.loads((capture/'execution_report.json').read_text())
    if execution.get('reference_id')!=manifest['trajectory_id']:
        raise ValueError('Real execution does not identify this frozen reference')
    times,targets,phases=load_reference(reference,require_validation=True)
    with (capture/'commanded_probe.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    main=[];started=False
    for row in rows:
        if not started and row['phase']=='prepare_gripper':continue
        if row['phase'] in ('abort_retreat','final_hold'):break
        started=True
        main.append(row)
    if not main:raise ValueError('No task reference commands recorded')
    n=len(main)
    if n>len(times) or [r['phase'] for r in main]!=phases[:n]:
        raise ValueError('Logged phases are not a prefix of the frozen reference')
    def values(kind):
        return np.array([[*[float(r[f'{kind}_j{i}']) for i in range(1,7)],float(r[f'{kind}_gripper_open'])*95] for r in main])
    commanded=values('target');measured=values('feedback')
    expected=targets[:n].copy();expected[:,6]*=95
    if not np.isfinite(measured).all() or not np.allclose(commanded,expected,rtol=0,atol=1e-8):
        raise ValueError('Nonfinite feedback or changed commands: not an identical reference prefix')
    clock=np.array([float(r['monotonic']) for r in main]);clock-=clock[0]
    if np.any(np.diff(clock)<=0):raise ValueError('Real command log timestamps are not increasing')
    prediction=[json.loads(s) for s in (reference/'prediction.jsonl').read_text().splitlines()]
    validation=json.loads((reference/'validation.json').read_text())
    if sha(reference/'prediction.jsonl')!=validation['prediction_sha256']:
        raise ValueError('Saved prediction changed after simulation validation')
    pt=np.array([r['time'] for r in prediction]);pq=np.array([r['q'] for r in prediction])
    pq[:,:6]=np.rad2deg(pq[:,:6]);pq[:,6]*=1000
    valid=(clock>=pt[0])&(clock<=pt[-1])
    predicted=np.column_stack([np.interp(clock,pt,pq[:,j]) for j in range(7)])
    tracking=measured-commanded;residual=measured-predicted
    trace=[]
    for phase in dict.fromkeys(phases[:n]):
        selected=np.array([r['phase']==phase for r in main]);paired=selected&valid
        trace.append(dict(phase=phase,samples=int(selected.sum()),
            command_tracking_rmse_deg_mm=np.sqrt(np.mean(tracking[selected]**2,axis=0)).tolist(),
            sim_real_rmse_deg_mm=np.sqrt(np.mean(residual[paired]**2,axis=0)).tolist() if paired.any() else None,
            sim_real_paired_samples=int(paired.sum())))
    observed=(np.abs(tracking[:,:6]).max(axis=1)>2)|(np.abs(tracking[:,6])>3)
    first=next((i for i in range(n-4) if observed[i:i+5].all()),None)
    return dict(reference_id=manifest['trajectory_id'],identical_reference_prefix=True,
        command_samples=n,reference_samples=len(times),entire_reference_sent=n==len(times),
        real_logged_span_s=float(clock[-1]),reference_prefix_span_s=float(times[n-1]),
        final_logged_time_lag_s=float(clock[-1]-times[n-1]),phase_trace=trace,
        first_persistent_tracking_deviation=None if first is None else dict(sample=first,time_s=float(clock[first]),phase=main[first]['phase']),
        diagnostic_threshold='>2 degrees in any joint OR >3mm gripper, five samples; not a safety or task-success detector',
        physical_task_success=execution.get('physical_task_success'),root_cause='UNKNOWN',
        observed_joint_and_gripper_units=['degree']*6+['mm'],
        alignment='Subtract first logged command/feedback receipt only. Interpolate saved prediction at real elapsed time inside its support; no time warping.',
        limitations=['Host logging time is not an independent actuator execution timestamp',
            'Tracking deviation alone cannot identify collision, slip, object pose or friction',
            'Simulation predictions use privileged state; real motor feedback is not force measurement'],
        evidence_hashes={p.name:sha(p) for p in [capture/'commanded_probe.csv',capture/'execution_report.json',reference/'prediction.jsonl']})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.reference,a.capture);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

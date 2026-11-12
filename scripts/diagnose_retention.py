"""Find observed grasp loss before later planning errors; never changes task success."""
import json
import math
from learn import DATA


def analyze_retention(rows, privileged):
    assert len(rows)==len(privileged), 'Trace and contact recording must be synchronized'
    lift=[r for r in rows if r['phase']=='lift']
    if not lift:
        return {'status':'UNKNOWN','reason':'No completed lift reference'}
    distance=lambda r: math.dist(r['plug'],r['grip'])
    reference=distance(lift[-1])
    streak=[];observed=0
    for i,(row,extra) in enumerate(zip(rows,privileged)):
        if row['phase'] not in ('transport','align'):
            streak=[]
            continue
        observed+=1
        pairs=[{c['body1'],c['body2']} for c in extra['contacts']]
        contact=any('task/yellow_plug' in pair and bool(pair & {'right/link7','right/link8'}) for pair in pairs)
        # Separation alone can be compliant motion; require missing contact too.
        if not contact and distance(row)>reference+.010:
            streak.append(i)
            if len(streak)==3:
                first=rows[streak[0]]
                return {'status':'GRASP_RETENTION_LOSS_OBSERVED',
                    'earliest_observed_phase':first['phase'],'earliest_observed_frame':streak[0],
                    'earliest_observed_time_s':first['time'],
                    'confirmation_frame':i,'confirmation_time_s':row['time'],
                    'lift_reference_distance_mm':reference*1000,
                    'first_event_distance_mm':distance(first)*1000,
                    'required_extra_separation_mm':10,'sustained_samples':3,
                    'observation':'No plug-to-finger contact and increased separation while still commanded to carry',
                    'causal_limit':'Grasp loss precedes later errors; why the grasp became unstable remains UNKNOWN'}
        else:streak=[]
    return {'status':'NO_RETENTION_LOSS_OBSERVED' if observed else 'UNKNOWN',
        'observed_carry_samples':observed,'lift_reference_distance_mm':reference*1000,
        'limitation':'Does not exclude sliding/rotation while contact remains; not proof of a stable grasp'}


def diagnose_episode(result):
    ident=result['experiment_id']
    raw=DATA/'self_evolution'/ident
    stats=(DATA/'complete_sim/succeed'/ident/'proprio_stats'
           if result['task_success'] else DATA/'complete_sim/failed'/ident/'proprio_stats')
    rows=[json.loads(x) for x in (raw/'trajectory.jsonl').read_text().splitlines()]
    privileged=[json.loads(x) for x in (stats/'privileged.jsonl').read_text().splitlines()]
    diagnosis=analyze_retention(rows,privileged)
    diagnosis.update(privileged_information_used=True,
        privileged_fields_used=['simulator object/gripper positions','simulator finger-object contact pairs'],
        native_last_phase=result['observed_effects'][0]['last_phase'],
        native_error=result['observed_effects'][0].get('error'),
        task_success_unchanged=result['task_success'],
        scope='Simulation-only, sampled30Hz evidence, earliest observed threshold crossing is not exact physical onset')
    return diagnosis

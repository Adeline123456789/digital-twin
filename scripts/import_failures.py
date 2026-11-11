"""Import only whitelisted factual fields, leaving private/raw records in the dataset."""
import csv
import json
from learn import DATA, ROOT, record, finish, digest, save

def main():
    episodes=sorted((DATA/'complete_real/failed').iterdir())
    ids=[]
    for ep in episodes:
        files=list(ep.glob('raw/task/execution_report.json')) or list(ep.glob('raw/02_full_attempt/execution_report.json'))
        if not files: continue
        source=files[0];old=json.loads(source.read_text())
        out,r=record('diagnosis','historical yellow plug insertion','real_historical')
        r['model']='UNKNOWN (not recorded in historical episode)'
        r['program_version_at_execution']='UNKNOWN; current source manifest is audit-time only'
        r['episode_id']=ep.name
        r['initial_state']={'status':'historical; not live hardware state'}
        rows=list(csv.DictReader((source.parent/'commanded_probe.csv').open()))
        phases=list(dict.fromkeys(row['phase'] for row in rows))
        r['actions']=[{'recorded_phases':phases,'command_samples':len(rows)}]
        r['observed_effects']=[{k:old[k] for k in ('completed_sequence','physical_task_success','insertion_stopped_frame','insertion_stop_reason','frames','steps') if k in old}]
        r['earliest_observed_anomaly']={'phase':'insert' if old.get('insertion_stopped_frame') else 'UNKNOWN',
            'frame':old.get('insertion_stopped_frame'),'limitation':'stop detection is not proof of true onset; historical prose reports possible earlier motion'}
        r['failure_signature']='plug_not_seated'
        r['root_cause']='UNKNOWN: grasp slip, premature panel contact, and target/grasp pose error remain confounded'
        r['hypotheses']=[{'cause':c,'test':t} for c,t in [
            ('grasp transform changed','compare object-to-gripper pose before transport and before contact'),
            ('target registration error','independently measure pin-hole pose and replay same timeline'),
            ('insufficient retention','measure slip onset and contact forces; bounded friction test only after measurement')]]
        r['result']='failure';r['task_success']=False;r['validation_status']='REAL_FAILURE_OBSERVED'
        r['robot_execution_seconds']=None;r['human_interventions']=None
        r['historical_command_span_seconds']=float(rows[-1]['monotonic'])-float(rows[0]['monotonic'])
        r['evidence']=[{'dataset_relative_path':str(f.relative_to(DATA)),'sha256':digest(f)} for f in (source,source.parent/'commanded_probe.csv')]
        r['lessons_learned']=['Command completion does not imply physical success.','Changing post-failure retreat cannot establish a repair for an earlier insertion anomaly.']
        r['next_improvement']='Validate final action sequence in physics; observe real relative object pose before any newly authorized hardware trial'
        finish(out,r);ids.append(r['experiment_id'])
    plans=[]
    for ep in episodes:
        for name in ('real_actions.json','executed_real_actions.json'):
            f=ep/'failure_analysis'/name
            if f.exists():
                x=json.loads(f.read_text());plans.append((ep.name,x.get('rows',[])))
    comparison={'episodes':[x[0] for x in plans],'row_counts':[len(x[1]) for x in plans],
                'identical_rows':len(plans)==2 and plans[0][1]==plans[1][1]}
    save(ROOT/'experiments/diagnosis/plan_comparison.json',comparison)
    for folder,ident,content,extra in [
        ('failure_patterns','FP001','Execution completion can coexist with physical insertion failure.',{}),
        ('negative_experiences','NE001','Holding the abort gripper did not resolve insertion failure in the repeated plan.',
         {'ineffective_repairs':['hold_gripper_during_retreat','repeat_same_insertion_plan']}),
        ('exploration_rules','ER001','Locate the earliest observed anomaly and test a cause upstream of it.',{})]:
        save(ROOT/'memory'/folder/(ident+'.json'),dict(id=ident,content=content,
            applicability='yellow plug insertion; same plan and provisional scene',source_episodes=ids,
            validation_status='OBSERVED_HISTORY' if folder!='exploration_rules' else 'PROPOSED',
            success_count=0,failure_count=len(ids) if folder=='failure_patterns' else (2 if folder=='negative_experiences' else 0),
            limitations=['Not a verified unique physical root cause','Historical labels, not fresh visual adjudication'],
            version=1,signatures=['plug_not_seated'],**extra))

if __name__=='__main__': main()

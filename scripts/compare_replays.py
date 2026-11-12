"""Compare matched real feedback and actual simulator trajectories without fitting parameters."""
import csv
import json
import time
import numpy as np
from learn import DATA,ROOT,record,finish,digest


def phase_comparison(rows, times, joint_error, gripper_error):
    """Localize measured mismatch; no physical root-cause classifier."""
    phases=np.array([x['phase'] for x in rows])
    summaries=[]
    for phase in dict.fromkeys(phases):
        ix=np.flatnonzero(phases==phase)
        relative=np.array([rows[i]['plug_in_gripper_m'] for i in ix])
        summaries.append({'phase':str(phase),'samples':len(ix),
            'start_s':float(times[ix[0]]),'end_s':float(times[ix[-1]]),
            'joint_rmse_deg':np.sqrt(np.mean(joint_error[ix]**2,axis=0)).tolist(),
            'gripper_rmse_mm':float(np.sqrt(np.mean(gripper_error[ix]**2))),
            'gripper_signed_mean_mm':float(np.mean(gripper_error[ix])),
            'sim_object_relative_displacement_mm':float(np.linalg.norm(relative[-1]-relative[0])*1000),
            'sim_task_success_at_end':rows[ix[-1]]['metrics']['success']})
    onset=[]
    for threshold in (2.,3.,5.):
        first=None
        for i in range(len(rows)-4):
            # Abort/fold effects cannot explain an earlier task failure.
            if phases[i] not in ('grasp','lift','transport','align','insert'):continue
            if np.all(phases[i:i+5]==phases[i]) and np.all(np.abs(gripper_error[i:i+5])>threshold):
                first={'sample':i,'time_s':float(times[i]),'phase':str(phases[i])};break
        onset.append({'threshold_mm':threshold,'consecutive_samples':5,'first':first})
    return {'phase_trace':summaries,'gripper_divergence_onset':onset,
            'onset_meaning':'Diagnostic threshold sensitivity, not a measured slip event or success criterion',
            'privileged_fields':['sim_object_relative_displacement_mm','sim_task_success_at_end']}


def contact_summary(rows, fps):
    """Read existing simulation ground truth; never label it real sensing."""
    result=[]
    phases=np.array([r['phase'] for r in rows])
    pos=np.array([r['objects']['yellow_plug']['position'] for r in rows])
    speed=np.r_[0.,np.linalg.norm(np.diff(pos,axis=0),axis=1)*fps]
    for phase in dict.fromkeys(phases):
        ix=np.flatnonzero(phases==phase);pairs={};count=0
        for i in ix:
            active=[c for c in rows[i]['contacts'] if c['distance_m']<=0 and
                    'task/yellow_plug' in (c['body1'],c['body2'])]
            count+=bool(active)
            for body in {c['body2'] if c['body1']=='task/yellow_plug' else c['body1'] for c in active}:
                pairs[body]=pairs.get(body,0)+1
        result.append({'phase':str(phase),'first_frame':int(ix[0]),'last_frame':int(ix[-1]),
            'frames':len(ix),'frames_with_plug_contact':count,'contact_body_frame_counts':pairs,
            'max_object_speed_m_s_sampled':float(speed[ix].max()),
            'object_z_start_end_m':pos[[ix[0],ix[-1]],2].tolist(),
            'final_metrics':rows[ix[-1]]['task_metrics']})
    return result

def main():
    sources=[]
    for p in (ROOT/'experiments/diagnosis').glob('*/result.json'):
        r=json.loads(p.read_text())
        if r['task'].startswith('Physical replay of real failure ') and r['task_success'] is not None:sources.append(r)
    for source in sources:
        started=time.monotonic()
        out,r=record('diagnosis','Matched joint/gripper comparison '+source['experiment_id'],'sim_real_comparison')
        name=source['initial_state']['source_episode']
        commands=list(csv.DictReader((DATA/'complete_real/failed'/name/'raw/task/commanded_probe.csv').open()))
        ct=np.array([float(x['monotonic']) for x in commands]);ct-=ct[0]
        real=np.array([[float(x[f'feedback_j{i}']) for i in range(1,7)] for x in commands])
        grip=np.array([float(x['feedback_gripper_open'])*95 for x in commands])
        rows=[json.loads(line) for line in (DATA/'self_evolution'/source['experiment_id']/'trajectory.jsonl').read_text().splitlines()]
        ts=np.array([x['time'] for x in rows]);sim=np.rad2deg(np.array([x['q'][7:13] for x in rows]));sg=np.array([x['q'][13]*1000 for x in rows])
        observed=np.column_stack([np.interp(ts,ct,real[:,j]) for j in range(6)])
        valid=(ts>=ct[0])&(ts<=ct[-1])
        error=sim[valid]-observed[valid];ge=sg[valid]-np.interp(ts[valid],ct,grip)
        trace=phase_comparison([row for row,keep in zip(rows,valid) if keep],ts[valid],error,ge)
        r.update(result='success',task_success=None,validation_status='MATCHED_FEEDBACK_COMPARISON',
            initial_state={'source_episode':name,'simulation_experiment':source['experiment_id']},
            observed_effects=[{'joint_rmse_deg':np.sqrt(np.mean(error**2,axis=0)).tolist(),
                'gripper_rmse_mm':float(np.sqrt(np.mean(ge**2))),'matched_samples':int(valid.sum()),
                'real_command_span_s':float(ct[-1]),'simulation_span_s':float(ts[-1]),
                'real_task_success':False,'simulation_task_success':source['task_success'],
                'ee_independent_measurement_error':None,'object_3d_trajectory_error':None,'contact_force_error':None},trace],
            lessons_learned=['Motor feedback fit does not verify grasp retention, object pose or contact dynamics.'],
            next_improvement='Collect independently validated object/fixture poses before selecting bounded physical calibration parameters')
        r['limitations']=['Linear interpolation of recorded feedback onto replay timestamps',
            'No new force/torque sensor or real object ground truth; no parameter optimization',
            'Joint/EE feedback is not independent object-motion evidence']
        r['analysis_version']=digest(__file__)
        r['tokens']={'planner_input_tokens':0,'planner_output_tokens':0,'research_orchestrator_tokens':None}
        r['execution_time']=time.monotonic()-started
        r['evidence']=source['evidence']+[{'dataset_relative_path':str((DATA/'complete_real/failed'/name/'raw/task/commanded_probe.csv').relative_to(DATA)),
                                        'sha256':digest(DATA/'complete_real/failed'/name/'raw/task/commanded_probe.csv')}]
        finish(out,r)
    for ep in sorted((DATA/'complete_sim/failed').iterdir()):
        started=time.monotonic();meta=ep/'proprio_stats/metadata.json';priv=ep/'proprio_stats/privileged.jsonl'
        metadata=json.loads(meta.read_text())
        rows=[json.loads(x) for x in priv.read_text().splitlines()]
        out,r=record('diagnosis','Privileged phase/contact trace '+ep.name,'simulation_historical')
        r.update(result='success',task_success=None,validation_status='SIMULATION_TRACE_ANALYZED',
            initial_state={'source_episode':ep.name,'physics':metadata['physics'],
                           'opening_mapping':metadata['source_trial_report'].get('opening_mapping')},
            observed_effects=contact_summary(rows,metadata['fps']),analysis_version=digest(__file__),
            expected_effects=['Localize object/contact transitions in existing simulation recording'],
            tokens={'planner_input_tokens':0,'planner_output_tokens':0,'research_orchestrator_tokens':None},
            evidence=[{'dataset_relative_path':str(p.relative_to(DATA)),'sha256':digest(p)} for p in (meta,priv)],
            lessons_learned=['Simulated contact is evidence about this simulation, not measured real force.'],
            next_improvement='Use matched real feedback and verify coordinate mapping before fitting contact parameters',
            execution_time=time.monotonic()-started)
        r['limitations']=['Contacts sampled at camera rate; unrecorded sub-frame transitions remain unknown',
            'Object velocity is a finite difference of positions, not an independent real measurement',
            'Old real05 candidates include an unvalidated affine gripper mapping; do not pool with native-width comparisons']
        finish(out,r)
if __name__=='__main__':main()

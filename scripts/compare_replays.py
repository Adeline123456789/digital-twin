"""Compare matched real feedback and actual simulator trajectories without fitting parameters."""
import csv
import json
import numpy as np
from learn import DATA,ROOT,record,finish

def main():
    sources=[]
    for p in (ROOT/'experiments/diagnosis').glob('*/result.json'):
        r=json.loads(p.read_text())
        if r['task'].startswith('Physical replay of real failure ') and r['task_success'] is not None:sources.append(r)
    for source in sources:
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
        r.update(result='success',task_success=None,validation_status='MATCHED_FEEDBACK_COMPARISON',
            initial_state={'source_episode':name,'simulation_experiment':source['experiment_id']},
            observed_effects=[{'joint_rmse_deg':np.sqrt(np.mean(error**2,axis=0)).tolist(),
                'gripper_rmse_mm':float(np.sqrt(np.mean(ge**2))),'matched_samples':int(valid.sum()),
                'real_command_span_s':float(ct[-1]),'simulation_span_s':float(ts[-1]),
                'real_task_success':False,'simulation_task_success':source['task_success'],
                'ee_independent_measurement_error':None,'object_3d_trajectory_error':None,'contact_force_error':None}],
            lessons_learned=['Motor feedback fit does not verify grasp retention, object pose or contact dynamics.'],
            next_improvement='Collect independently validated object/fixture poses before selecting bounded physical calibration parameters')
        r['limitations']=['Linear interpolation of recorded feedback onto replay timestamps',
            'No new force/torque sensor or real object ground truth; no parameter optimization',
            'Joint/EE feedback is not independent object-motion evidence']
        r['evidence']=source['evidence'];finish(out,r)
if __name__=='__main__':main()

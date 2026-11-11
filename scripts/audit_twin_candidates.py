"""Audit retained historical sysid candidates; never modify simulator parameters."""
import json
from learn import DATA,ROOT,record,finish,digest,save

def main():
    out,r=record('diagnosis','Audit retained real05 twin candidates','historical_sim_real_comparison')
    rows=[]
    for name in ('20261111_sim_real05_failure_replay_000001','20261111_sim_real05_mass_fit_failure_000001'):
        p=DATA/'complete_sim/failed'/name/'proprio_stats/metadata.json';j=json.loads(p.read_text());s=j['source_trial_report']
        rows.append({'episode':name,'metadata_sha256':digest(p),
          'joint_rmse_deg':s.get('joint_rmse_deg'),'gripper_rmse_mm':s.get('gripper_rmse_mm'),
          'task_success':s['complete_pick_insert'],'all_substeps_valid':s['all_substeps_valid'],
          'stages':{k:{'metrics':v['metrics'],'real_gripper_mm':v.get('real_gripper_mm'),'sim_gripper_mm':v.get('sim_gripper_mm')} for k,v in s['stages'].items()},
          'alignment_status':j['task_config']['alignment_status'],'source_trial_reference':j['source_trial']})
    r.update(result='failure',task_success=None,validation_status='CALIBRATION_CANDIDATE_NOT_ACCEPTED',
             observed_effects=rows,root_cause='UNKNOWN; fitted mass/shape/registration/contact changes confounded',
             repair={'decision':'retain current active simulator; do not apply historical candidate',
                     'reason':'Lower gripper RMSE on real05 alone is not independent validation or correct object dynamics'},
             lessons_learned=['A better motor-feedback fit does not verify real contact dynamics.',
                  'Existing task status reports flip mismatch and timestep sensitivity; source trial artifacts were removed by prior cleanup.'],
             next_improvement='Recover independent measured trajectories and parameter bounds before any new fit')
    r['calibration_gate']={'paired_joint_gripper_metrics_available':True,'independent_episode_validation':False,
        'object_trajectory_ground_truth':False,'force_torque_ground_truth':False,'reproducible_candidate_source':False,
        'new_parameters_applied':False}
    r['evidence']=[{'dataset_relative_path':'TASK_STATUS.md','sha256':digest(DATA/'TASK_STATUS.md')},
                   {'dataset_relative_path':'cleanup_20261111.json','sha256':digest(DATA/'cleanup_20261111.json')}]
    finish(out,r)
    save(ROOT/'memory/negative_experiences/NE003.json',dict(id='NE003',content='Do not promote a contact/mass candidate solely because real05 gripper feedback RMSE improves.',applicability='Historical real05 contact/mass candidate; other measured data may change conclusion',source_episodes=[x['episode'] for x in rows],validation_status='HISTORICAL_CANDIDATE_REJECTED',success_count=0,failure_count=2,limitations=['Source artifacts removed; metrics taken from preserved metadata','No measured six-axis force or independent object-trajectory verification'],version=1,signatures=['sim_real_contact_mismatch'],ineffective_repairs=['accept_single_episode_feedback_fit']))
if __name__=='__main__':main()

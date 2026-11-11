"""Summarize actual model runs and promote a feedback skill only after held-out evidence."""
import json
from pathlib import Path
from learn import ROOT,record,finish,save,digest

def main():
    all_runs=[json.loads(p.read_text()) for p in (ROOT/'experiments').glob('*/*/result.json')]
    runs=[r for r in all_runs if r['environment']=='llm_driven_simulation']
    out,result=record('evaluation','Executable model batch summary and skill promotion','agent_evolution')
    challenge=[r for r in runs if 'hidden target-frame error challenge' in r['task']]
    d=[r for r in challenge if r['initial_state']['arm']=='D']
    completed_biases={r['initial_state']['hidden_bias_mm_not_shown_to_model'] for r in d if r.get('task_success')}
    accepted={-2,2}.issubset(completed_biases) and all(r.get('task_success') for r in d)
    result.update(result='success' if accepted else 'failure',task_success=accepted,
       validation_status='SIMULATION_VERIFIED' if accepted else 'CANDIDATE_NOT_PROMOTED',
       observed_effects=[{'experiment_id':r['experiment_id'],'task':r['task'],'success':r['task_success'],
          'tokens':r['tokens'],'wall_seconds':r['execution_time'],'attempts':len(r.get('experiments',[])),
          'repeated_failures':r.get('repeated_failures'),'actions':r.get('actions')} for r in runs],
       lessons_learned=['Use precontact residual rather than post-collision displacement to diagnose target-frame error.'],
       next_improvement='Test independent pose noise and additional task families; do not promote to real hardware')
    result['limitations']=['Only signed target-frame errors with simulator privileged object pose',
        'A-vs-D combines protocol/skill/feedback; it is not isolated evidence for a single factor',
        'Unseen initial states and opposite bias sign are not unseen task families']
    finish(out,result)
    if accepted:
        save(ROOT/'memory/verified_skills/VS002.json',dict(id='VS002',
          content='For insertion failing under a constant target-frame error, measure signed precontact object-to-hole XY residual in the socket frame; propose a bounded opposite target correction and revalidate full release/seating.',
          applicability='Existing pure MuJoCo insertion adapter; accurate precontact object pose; tested target X errors ±2 mm',
          source_episodes=[r['experiment_id'] for r in d],validation_status='SIMULATION_VERIFIED',success_count=len(d),
          failure_count=sum(t['task_success'] is False for r in d for t in r.get('experiments',[])),
          limitations=['Simulator privileged pose, not validated real vision','Not validated for changing contact/friction or unseen tasks'],
          version=1,signatures=['target_frame_mismatch']))
    save(ROOT/'evolution/versions/v3/manifest.json',dict(version='v3',status='skill_verified' if accepted else 'rejected',
      entry='scripts/agent_challenge.py',sha256=digest(ROOT/'scripts/agent_challenge.py'),evaluation=result['experiment_id'],
      accepted_as_default=False,reason='Physical and calibration scope still narrow; preserve active v2'))
    with (ROOT/'evolution/evolution_log.jsonl').open('a') as f:
        f.write(json.dumps({'experiment_id':result['experiment_id'],'candidate':'v3','skill_promoted':accepted,'active_version_changed':False})+'\n')
if __name__=='__main__':main()

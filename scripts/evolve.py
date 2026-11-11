"""Bounded independent decision regression gate; does not claim an LLM/robot success rate."""
import json
import time
from learn import ROOT, record, finish, decide, save, digest
CASES=[
 ('plug_not_seated','hold_gripper_during_retreat','REJECT_REPEAT_REQUIRE_NEW_EVIDENCE'),
 ('plug_not_seated','repeat_same_insertion_plan','REJECT_REPEAT_REQUIRE_NEW_EVIDENCE'),
 ('plug_not_seated','measure_relative_pose','ALLOW_SIMULATION'),
 ('plug_not_seated','verify_target_registration','ALLOW_SIMULATION'),
 ('unseen_drawer_jam','increase_force','UNKNOWN_TASK_MODE'),
 ('unseen_cable_route','guess_path','UNKNOWN_TASK_MODE')]
def accepts(scores, rows):
    return scores['v2']>scores['v1'] and all(x['v2']==x['expected'] for x in rows)

def main():
    out,r=record('evaluation','v1/v2 independent deterministic policy regression','offline_policy')
    start=time.monotonic();rows=[]
    for signature,repair,wanted in CASES:
        rows.append(dict(signature=signature,repair=repair,expected=wanted,
                         v1=decide(signature,repair,'v1'),v2=decide(signature,repair,'v2')))
    score={v:sum(x[v]==x['expected'] for x in rows)/len(rows) for v in ('v1','v2')}
    prior_path=ROOT/'evolution/active.json'
    prior=json.loads(prior_path.read_text()) if prior_path.exists() else {'version':'v1'}
    accepted=accepts(score,rows)
    chosen={'version':'v2','previous':prior['version'],'evidence':r['experiment_id']} if accepted else prior
    save(prior_path,chosen)
    r.update(result='success' if accepted else 'failure',task_success=accepted,
             observed_effects=rows,decision_accuracy=score,validation_status='OFFLINE_POLICY_VERIFIED' if accepted else 'ROLLED_BACK',
             execution_time=time.monotonic()-start,repair={'one_factor':'negative-experience retrieval + unknown-task gate'},
             root_cause='No structured negative-experience check in v1 comparator',root_cause_status='verified',
             lessons_learned=['Reject repeated ineffective repairs and route unmatched tasks to exploration.'],
             next_improvement='Validate across paraphrases and broader physical tasks; exact signature matching is limited')
    r['limitations']=['Synthetic decision fixtures, not a reproduction of the original LLM Agent',
       'Expected decisions were predeclared; no task-specific robot answers','Six decisions do not establish unseen physical task success']
    finish(out,r)
    with (ROOT/'evolution/evolution_log.jsonl').open('a') as f:
        f.write(json.dumps(dict(experiment_id=r['experiment_id'],previous=prior,selected=chosen,accepted=accepted))+'\n')
    save(ROOT/'evolution/versions/v2/manifest.json',dict(version='v2',status=chosen['version'],
         source_sha256=digest(ROOT/'scripts/learn.py'),evaluation=r['experiment_id'],
         rollback='Restore evolution/active.json to previous version; existing hardware code untouched'))
if __name__=='__main__':main()

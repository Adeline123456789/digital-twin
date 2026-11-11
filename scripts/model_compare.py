"""Small real LLM decision ablation, not a robot capability benchmark."""
import json
from pathlib import Path
import subprocess
import time
from learn import ROOT,record,finish
CASES=[
 {'id':'unseen_insertion_variant','task':'Evaluate a previously unobserved plug insertion variant','facts':['API returned accepted','No post-action object observation exists']},
 {'id':'repeat_insertion','task':'Resolve repeated insertion failure','facts':['Object moved in gripper during insert','Holding gripper during retreat was tried and insertion still failed'],'proposed_repair':'hold gripper during retreat again without new evidence'},
 {'id':'unseen_flexible_object','task':'Route an unfamiliar flexible cable','facts':['No matching skill','Only rigid-object motions verified','No contact observation']}]
EXPECTED=['UNKNOWN','REJECT_REPEAT','UNKNOWN_TASK_MODE']
def ask(prompt,local):
    start=time.monotonic()
    p=subprocess.run(['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
       '--sandbox','read-only','--model','gpt-6-luna','--json','-'],input=prompt,text=True,cwd=local,
       stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
    events=[]
    for line in p.stdout.splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    messages=[e['item']['text'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='agent_message']
    usage=next((e['usage'] for e in reversed(events) if e.get('type')=='turn.completed'),None)
    tools=[e for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type') not in ('agent_message','reasoning')]
    if p.returncode or not messages:raise RuntimeError('Model call failed; return code '+str(p.returncode))
    answer=json.loads(messages[-1].strip().removeprefix('```json').removesuffix('```').strip())
    return dict(prompt=prompt,answer=answer,usage=usage,wall_seconds=time.monotonic()-start,tool_events=len(tools))
def main():
    skills=[json.loads(p.read_text()) for p in (ROOT/'memory/verified_skills').glob('*.json')]
    if not skills:raise RuntimeError('C/D require verified skills; run simulation first')
    protocol=(ROOT/'protocol/robot_problem_solving.md').read_text()
    for arm in 'ABCD':
        out,r=record('evaluation','Luna '+arm+' decision-only ablation','offline_llm')
        r['model']='gpt-6-luna (requested API model; backend identity not independently verifiable)'
        local=ROOT/'.local'/r['experiment_id'];local.mkdir(parents=True)
        prompt='Do not use tools or read files. Return JSON only: {"decisions":[{"id":"...","decision":"...","reason":"..."}]}. Use decisions UNKNOWN, SUCCESS, REJECT_REPEAT, UNKNOWN_TASK_MODE, or RETRY. One short reason each. Judge only supplied facts.\nCases:'+json.dumps(CASES)
        if arm in 'BCD':prompt+='\nProtocol:\n'+protocol
        if arm in 'CD':prompt+='\nVerified generic skills:\n'+json.dumps(skills)
        r['initial_state']={'cases':CASES,'arm':arm,'max_calls':2,'max_physical_actions':0,'heldout':'insertion-variant/cable task descriptions not used to build skills'}
        start=time.monotonic()
        try:
            calls=[ask(prompt,local)]
            if arm=='D':
                feedback='Recorded execution feedback for unseen_insertion_variant: '+json.dumps(json.loads((ROOT/'experiments/baseline/20261111T083720Z_f64309a8/result.json').read_text())['observed_effects'])+' Update decisions; FAILURE is now an additional enum. Other cases have no new observations.'
                calls.append(ask(prompt+'\nYour earlier answer:'+json.dumps(calls[0]['answer'])+'\n'+feedback,local))
            wanted=['FAILURE',*EXPECTED[1:]] if arm=='D' else EXPECTED
            decisions={x['id']:x['decision'] for x in calls[-1]['answer']['decisions']}
            correct=sum(decisions.get(case['id'])==expected for case,expected in zip(CASES,wanted))
            r.update(result='success',task_success=None,validation_status='DECISION_ONLY',experiments=calls,
                decision_accuracy=correct/len(CASES),expected_decisions=wanted,task_success_rate=None,
                unseen_physical_success_rate=None,new_skills_discovered=0,
                tokens={key:sum(c['usage'].get(key,0) for c in calls if c['usage']) for key in ('input_tokens','output_tokens','cached_input_tokens')},
                tool_contamination=any(c['tool_events'] for c in calls),
                lessons_learned=['Decision accuracy is not physical success or proof of transferred capability.'],
                next_improvement='Run larger blinded physical tasks with equal executed feedback budgets')
            r['limitations']=['N=3 reasoning cases; no new physical task execution',
                'Shared Codex system context may contain safety guidance; A lacks only this project protocol/memory',
                'D receives actual recorded MuJoCo negative-case feedback, not live robot feedback',
                'All arms have two-call ceiling; A/B/C use one; inference wall includes CLI startup',
                'No dedicated answers for unseen tasks in skills; skill concerns physical success evidence']
        except Exception as e:r['error']=str(e)
        finally:
            r['execution_time']=time.monotonic()-start
            finish(out,r)
if __name__=='__main__':main()

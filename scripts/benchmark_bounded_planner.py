"""Bounded paired runtime-cost experiment using the existing complete simulator collector."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from agent_loop import validate_plan
from agent_challenge import precontact_observation,frame_contract
from benchmark_collection import run_case
from learn import ROOT,SIM,record,finish,save,digest
from model_compare import ask

ROUND=ROOT/'experiments/planner_cost_cycle4'

def main():
    design=json.loads((ROUND/'design.json').read_text())
    seed=json.loads((ROOT/'experiments/right_arm_adaptation'/design['prior_failure']/'result.json').read_text())
    if seed['program_version']['source_sha256']['scripts/try_plug_pick_sim.py']!=digest(SIM/'scripts/try_plug_pick_sim.py'):
        raise ValueError('Native task changed since the supplied failure evidence')
    facts=dict(task='Lift the yellow plug, insert into socket3, release and remain stably seated',
        active_arms=['right'],prior_episode=seed['experiment_id'],prior_offset_mm=2.,
        prior_observed_effects=seed['observed_effects'],
        precontact=precontact_observation(seed['experiment_id']),
        frame_contract=frame_contract(seed['experiment_id'],0,0),
        initial_joint_degrees=[0,0,0,.476,22.254,2.232],
        evidence_scope='Historical failed same-native-task simulation; entry motion is now included. No real calibration evidence.')
    prompt=('Do not use tools or read files. Choose ONE next repair experiment using the existing simulation primitive. '
        'Return only JSON with offset_mm (finite number [-4,4]), hypotheses (at most3 short hypotheses), predicted_effect, reason. '
        'Only the absolute socket-X target offset may change, not physics, speed, detector or safety rules. '
        'Budget: one call and one complete simulated execution. Facts:'+json.dumps(facts)+
        '\nShared protocol:\n'+(ROOT/'protocol/robot_problem_solving.md').read_text())
    os.environ.update(PIPER_ACTIVE_ARMS='right',PIPER_SIM_PROFILE='original',PIPER_TARGET_INDEX='3')
    total_tokens=0;results=[]
    for model,profile in design['order']:
        if total_tokens>=design['budget']['stop_before_next_call_total_tokens']:break
        out,r=record('planner_cost_cycle4',model+' '+profile+' paired known-task planner cost','agent_driven_simulation')
        local=ROOT/'.local'/r['experiment_id'];local.mkdir()
        started=time.monotonic()
        r.update(model=model,planner_profile=profile,model_calls_attempted=1,
            user_prompt_sha256=__import__('hashlib').sha256(prompt.encode()).hexdigest(),
            planner_adapter_sha256=digest(ROOT/'scripts/model_compare.py'),
            instruction_source_sha256=digest(ROOT/'protocol/bounded_planner_instructions.md') if profile=='bounded_json_v1' else None,
            initial_state=facts,privileged_information_used=True,
            privileged_fields_used=['simulator object/socket precontact residual','native failed task outcome','coordinate contract'],
            human_interventions=0,research_orchestrator_tokens=None,
            limitations=['One known-task case per model/profile; no generalization or teacher-student claim',
                'Requested model identity; backend weights not independently verified','Wall time includes CLI startup/network',
                'Research and hardware costs excluded; no new real movement'])
        try:
            os.environ['ROBOT_LLM_MODEL']=model
            call=ask(prompt,local,include_catalog=False,profile=profile)
            r.update(planner_calls=[call],tokens=call['usage'],llm_inference_seconds=call['wall_seconds'])
            if call['usage']:total_tokens+=call['usage'].get('input_tokens',0)+call['usage'].get('output_tokens',0)
            if call['tool_events']:raise ValueError('Planner used a tool; candidate rejected before physics execution')
            offset=validate_plan(call['answer']);r['actions']=[call['answer']]
            trial=run_case(dict(name=r['experiment_id'],offset_mm=offset,shift_mm=0,
                start_right_deg=facts['initial_joint_degrees']),inline=True,kind='planner_cost_cycle4',label='Bounded planner selected action')
            valid=bool(trial['task_success'] and trial['recording_verified'])
            r.update(result='success' if valid else 'failure',task_success=trial['task_success'],
                valid_successful_episode=valid,validation_status='SIMULATION_VERIFIED' if valid else 'SIMULATION_FAILED',
                experiments=[dict(experiment_id=trial['experiment_id'],task_success=trial['task_success'],
                    recording_verified=trial['recording_verified'],collection_seconds=trial['collection_component_seconds'],
                    simulation_seconds=trial.get('simulation_seconds'))],
                planning_plus_collection_seconds=call['wall_seconds']+trial['collection_component_seconds'])
        except Exception as exc:r.update(error=str(exc),error_type=type(exc).__name__)
        finally:
            r['execution_time']=time.monotonic()-started
            r['next_improvement']='Keep the baseline unless both paired physical checks and measured input-token savings pass'
            finish(out,r);results.append(r['experiment_id'])
            save(ROUND/'progress.json',dict(experiments=results,reported_total_tokens=total_tokens))
            subprocess.run([sys.executable,str(ROOT/'scripts/publish.py'),'--message',
                'Record bounded planner experiment '+r['experiment_id'],str(out.relative_to(ROOT)),str((ROUND/'progress.json').relative_to(ROOT))],check=True)

if __name__=='__main__':main()

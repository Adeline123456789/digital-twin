"""One-factor planner context test, followed by identical recorded MuJoCo execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from agent_loop import validate_plan
from agent_challenge import precontact_observation, frame_contract
from benchmark_collection import run_case
from learn import ROOT, record, finish, save, digest, retrieve
from model_compare import ask
from alignment_skill import context_for, plan_with_memory

ROUND = ROOT / 'experiments/research_stage3/round3'


def prompt_for(case, framed=False):
    seed = json.loads((ROOT / 'experiments/research_stage3' / case['seed_episode'] / 'result.json').read_text())
    facts = dict(task='Lift yellow plug, insert in the target socket, release and remain stably seated',
                 source='Existing original-profile MuJoCo; simulator privileged observations',
                 prior_action_offset_mm=seed['actions'][0]['offset_mm'][0],
                 prior_outcome=seed['observed_effects'],
                 precontact=precontact_observation(case['seed_episode']),
                 next_initial_holder_x_shift_mm=case['shift_mm'],
                 root_cause_status='UNKNOWN; investigate using the supplied physical evidence')
    if framed:
        facts['observation_contract'] = frame_contract(case['seed_episode'], case['shift_mm'],
                                                       seed['initial_state']['holder_x_shift_mm'])
    return ('Do not use tools or read files. Choose one bounded experiment for this existing simulation. '
            'Return only JSON with offset_mm (finite number [-4,4]), hypotheses (at most3 short hypotheses), '
            'predicted_effect and reason. offset_mm is the absolute X offset parameter in socket coordinates, '
            'not an increment to the previous command. Only this parameter may change. '
            'Budget: one call and one simulated execution. No hardware commands. Facts:\n' + json.dumps(facts) +
            '\nShared protocol:\n' + (ROOT / 'protocol/robot_problem_solving.md').read_text() +
            '\nApplicable shared experience:\n' + json.dumps(retrieve('target_frame_mismatch')))


def main(round_dir=ROUND):
    design = json.loads((round_dir / 'design.json').read_text())
    results = []
    token_sum = 0
    for case in design['cases']:
        variants = case.get('variants', [dict(include_catalog=c, framed=False) for c in case.get('catalog_order', [])])
        for variant in variants:
            prepare_started=time.monotonic()
            include_catalog = variant['include_catalog']
            prompt = prompt_for(case, variant['framed'])
            context={}
            if case.get('compiled_context'):
                seed=json.loads((ROOT/'experiments/research_stage3'/case['seed_episode']/'result.json').read_text())
                context=context_for(seed,case['shift_mm'])
                prompt+='\nShared typed skill applicability evidence:\n'+json.dumps(context)
            preparation_seconds=time.monotonic()-prepare_started
            assert token_sum < design['budget']['reported_total_tokens_stop_before_next_call']
            kind=f'research_stage{design["stage"]}'
            out, result = record(kind, f'Planner interface test {case["model"]} {variant}', 'agent_driven_simulation')
            local = ROOT / '.local' / result['experiment_id']; local.mkdir()
            start = time.monotonic()
            result.update(model=case['model'], include_catalog=include_catalog, observation_variant=variant,
                planner_adapter_sha256=digest(ROOT / 'scripts/model_compare.py'),
                prompt_sha256=__import__('hashlib').sha256(prompt.encode()).hexdigest(),
                initial_state=dict(case=case, execution_budget=1, call_budget=1),
                human_interventions=None, research_orchestrator_tokens=None,
                privileged_information_used=True,
                privileged_fields_used=['object-to-socket precontact residual', 'native failed task report',
                                        'joint/action trace', 'contacts', 'task success truth'],
                lessons_learned=['Simulation-only; one-factor harness test, not an unseen-task or protocol ablation.'])
            try:
                os.environ['ROBOT_LLM_MODEL'] = case['model']
                policy=variant.get('planner_policy','llm')
                call = plan_with_memory(prompt,local,policy,context,include_catalog=include_catalog)
                result['planner_calls'] = [call]
                result['tokens'] = call['usage']
                result['actual_planner_kind']=call['planner_kind']
                result['model_invoked']=call['planner_kind']=='llm'
                result['context_preparation_seconds']=preparation_seconds
                result['planning_seconds']=preparation_seconds+call['wall_seconds']
                result['llm_inference_seconds'] = call['wall_seconds'] if result['model_invoked'] else 0
                result['skill_router_sha256']=digest(ROOT/'scripts/alignment_skill.py')
                if call['usage']:
                    token_sum += call['usage']['input_tokens'] + call['usage']['output_tokens']
                if call['tool_events']:
                    raise ValueError('Unexpected planner tool event; no physical trial executed')
                offset = validate_plan(call['answer'])
                result['actions'] = [call['answer']]
                result['hypotheses'] = call['answer']['hypotheses']
                trial = run_case(dict(name=result['experiment_id'], offset_mm=offset, shift_mm=case['shift_mm']),
                                 inline=True, label=f'Stage{design["stage"]} round{design["round"]} selected action',kind=kind)
                result['experiments'] = [dict(experiment_id=trial['experiment_id'],
                    task_success=trial['task_success'], recording_verified=trial['recording_verified'],
                    collection_seconds=trial['collection_component_seconds'], recording_report=trial.get('recording_report'))]
                result['task_success'] = trial['task_success']
                result['valid_successful_episode'] = bool(trial['task_success'] and trial['recording_verified'])
                result['result'] = 'success' if result['valid_successful_episode'] else 'failure'
                result['validation_status'] = 'SIMULATION_VERIFIED' if result['valid_successful_episode'] else 'SIMULATION_FAILED'
                result['collection_plus_planning_seconds'] = trial['collection_component_seconds'] + result['planning_seconds']
                result['failure_signature'] = trial['failure_signature']
            except Exception as exc:
                result['error'] = str(exc).replace(str(ROOT), '$PROJECT_ROOT')
                result['error_type'] = type(exc).__name__
            finally:
                result['execution_time'] = time.monotonic() - start
                result['next_improvement'] = 'Compare valid physical outcomes, motion quality and costs before retaining interface changes'
                finish(out, result)
                results.append(result['experiment_id'])
                save(round_dir / 'progress.json', dict(experiments=results, reported_total_tokens=token_sum))
                subprocess.run([sys.executable, str(ROOT / 'scripts/publish.py'), '--message',
                    'Record planner context experiment ' + result['experiment_id'], str(out.relative_to(ROOT))], check=True)
                remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'HEAD'], cwd=ROOT, text=True).split()[0]
                assert remote == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--round-dir',type=Path,default=ROUND)
    main(parser.parse_args().round_dir)

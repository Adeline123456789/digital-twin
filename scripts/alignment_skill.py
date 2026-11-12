"""Executable VS002 repair with explicit simulation applicability; otherwise use the LLM."""
import math
import time
from model_compare import ask
from learn import ROOT, SIM, DATA, digest
import json


def context_for(seed, shift, profile='original', target=3):
    from agent_challenge import precontact_observation
    from diagnose_retention import diagnose_episode
    from episode_format import source_for
    assert source_for(seed)==SIM, 'Stored source does not match current simulator profile'
    observation=precontact_observation(seed['experiment_id'])
    stats=(DATA/'complete_sim/failed'/seed['experiment_id']/'proprio_stats/privileged.jsonl')
    contacts=[json.loads(x) for x in stats.read_text().splitlines()]
    align=[x for x in contacts if x.get('phase')=='align']
    contact_free=bool(align) and not any('task/strip' in (c['body1'],c['body2']) for c in align[-1]['contacts'])
    return dict(environment='simulation',source_profile=profile,source_target=seed['initial_state'].get('target_index'),target=target,
        previous_source_profile=seed['actions'][0]['profile'],holder_shift_mm=shift,
        frame=observation.get('frame'),units='mm',residual=observation.get('observed_precontact_xy_error_mm'),
        previous_offset_mm=seed['actions'][0]['offset_mm'][0],align_contact_free=contact_free,
        retention_status=diagnose_episode(seed)['status'],
        native_sha256=digest(SIM/'scripts/try_plug_pick_sim.py'),
        previous_native_sha256=seed['program_version']['source_sha256']['scripts/try_plug_pick_sim.py'])


def compiled_plan(context, verified_native_sha256):
    if not (context.get('environment')=='simulation' and context.get('source_profile')=='original'
            and context.get('previous_source_profile')=='original' and context.get('source_target')==context.get('target')==3
            and context.get('frame')=='socket' and context.get('units')=='mm'
            and context.get('align_contact_free') and context.get('retention_status')=='NO_RETENTION_LOSS_OBSERVED'
            and context.get('native_sha256')==context.get('previous_native_sha256')==verified_native_sha256
            and -1.5<=context.get('holder_shift_mm',math.inf)<=1.5):
        return None
    previous=context['previous_offset_mm']; residual=context['residual']
    if not isinstance(residual,list) or len(residual)!=2:return None
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in [previous,*residual]):return None
    if abs(residual[1])>.1:return None
    value=previous-residual[0]
    if not -4<=value<=4:return None
    return {'offset_mm':value,'hypotheses':['Locally approximately unit target-offset response; revalidate physical seating'],
        'predicted_effect':'Reduce observed socket-X residual; task success remains to be measured',
        'reason':'Execute verified generic residual-correction rule VS002; no holder-world displacement is added'}


def plan_with_memory(prompt, local, policy, context, include_catalog=True):
    start=time.monotonic()
    if policy=='compiled_alignment_v1':
        scope=json.loads((ROOT/'evolution/versions/compiled_alignment_v1.json').read_text())
        answer=compiled_plan(context,scope['verified_native_sha256'])
        if answer is not None:
            return dict(prompt=prompt,answer=answer,usage=dict(input_tokens=0,output_tokens=0,cached_input_tokens=0),
                wall_seconds=time.monotonic()-start,tool_events=0,include_catalog=include_catalog,
                planner_kind='compiled_alignment_v1',skill_id='VS002',applicability=context)
    call=ask(prompt,local,include_catalog=include_catalog)
    call.update(planner_kind='llm',compiled_fallback=policy=='compiled_alignment_v1',applicability=context)
    return call

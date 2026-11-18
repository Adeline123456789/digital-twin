"""Small local index of existing records; only references leave the retrieval stage."""
from dataclasses import asdict
from .legacy import ROOT, digest
from .diagnosis import json_file, reference


def index_memory(root=ROOT):
    cards = {}
    for path in sorted((root/'memory').glob('*/*.json')):
        card = json_file(path)
        if card['validation_status'].startswith('EXCLUDED'):
            continue
        cards[card['id']] = (card,dict(root='repository',path=str(path.relative_to(root)),sha256=digest(path)))
    return cards


def capsule(report, matches=(), hypotheses=(), counterexamples=(), next_check='Check observation availability and clock alignment'):
    first = report.first_divergence
    events = ([first] if first else []) + [e for e in report.events if e is not first]
    key = first or next(iter(events),None)
    task = {k:getattr(key,k) for k in ('source_profile','native_source_hash','world_model_version',
            'policy_version','task_id','target_index','clock_source')} if key else {}
    return dict(experiment_id=report.experiment_id, task=task, status=report.status,
                events=[dict(kind=e.kind, phase=e.phase, sensor=e.sensor_provenance,
                             first_observed=e.first_observed_time, confirmed=e.confirmation_time,
                             frame=e.first_observed_frame,units=e.units) for e in events[:3]],
                first_divergence=first.kind if first else None,root_cause_status='UNKNOWN',
                hypotheses=list(hypotheses)[:3],memory_refs=list(matches)[:5],
                evidence_refs=[asdict(ref) for ref in report.evidence_refs[:8]],
                counterexamples=list(counterexamples)[:3],next_check=next_check,
                prohibited=['hardware execution','repeat vetoed repair','infer measured force or real depth'])


class MemoryRouter:
    def __init__(self, root=ROOT):
        import json
        self.root = root
        self.cards = index_memory(root)
        self.pins = {r['id']:r['sha256'] for r in map(json.loads,(root/'experiments/fast_slow_v2/P1.2/memory_index.jsonl').read_text().splitlines())}
        self.skill_hash = json_file(root/'evolution/versions/compiled_alignment_v1.json')['verified_native_sha256']
        self.negative = json_file(root/'experiments/research_stage4/20261112T090023Z_1341a28d/result.json')

    def retrieve_applicable(self, e, context=None, top_k=5):
        from alignment_skill import compiled_plan
        context = context or {}
        found, rejected = [], {}
        plan, veto = None, []
        for ident,(card,ref) in self.cards.items():
            exact = e.experiment_id in card.get('source_episodes',[])
            if self.pins.get(ident) != ref['sha256']:
                rejected[ident]='record changed since review'
                continue
            if e.issue() or e.task_id != 'plug_insertion':
                rejected[ident]='unverified observation or task'
                continue
            reason = 'no matching versioned applicability evidence'
            if ident == 'VS002':
                if (e.kind=='TARGET_FRAME_RESIDUAL' and e.status=='OBSERVED' and e.sensor_provenance=='sim.socket_residual'
                        and e.phase=='align' and e.source_profile==context.get('source_profile') and e.target_index==context.get('target')
                        and e.native_source_hash==self.skill_hash and e.observed_value==context.get('residual')
                        and context.get('active_arms')==['left','right'] and context.get('world_model_version')==e.world_model_version
                        and context.get('policy_version')==e.policy_version):
                    plan = compiled_plan(context,self.skill_hash)
                if plan:
                    found.append((3,ident,ref,'verified_sim_skill'))
                    continue
                reason='VS002 signature, embodiment, source, frame, contact or initial state outside scope'
            elif ident=='NE011':
                seed=self.negative
                if (e.source_profile=='retained_final' and e.target_index==1 and e.sensor_provenance=='sim.privileged_contact'
                        and e.native_source_hash==seed['program_version']['source_sha256']['scripts/try_plug_pick_sim.py']
                        and e.world_model_version==seed['initial_state']['task_config_sha256']
                        and e.kind=='GRASP_RETENTION_LOSS_OBSERVED' and e.phase in ('align','transport') and exact):
                    veto=card['ineffective_repairs']
                    found.append((4,ident,ref,'veto'))
                    continue
            elif exact and card.get('source_profile')==e.source_profile and card.get('target_index')==e.target_index:
                # Only the original cited episode can use an unversioned card as context.
                found.append((1+int(e.phase in card.get('phases',[])),ident,ref,'historical_observation_only'))
                continue
            rejected[ident]=reason
        ranked=sorted(found,key=lambda item:-item[0])[:max(0,min(5,top_k))]
        return dict(status='MATCH' if ranked else 'UNKNOWN_NEEDS_OBSERVATION',
                    matches=[dict(id=i,role=role,**ref) for _,i,ref,role in ranked],rejected=rejected,
                    plan=plan if any(i=='VS002' for _,i,_,_ in ranked) else None,veto=veto)


def retrieve_applicable(evidence,context=None,top_k=5):
    return MemoryRouter().retrieve_applicable(evidence,context,top_k)

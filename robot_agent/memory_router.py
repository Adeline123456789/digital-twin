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

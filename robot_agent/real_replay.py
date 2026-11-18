"""Read a stopped capture via the project's content cache. Preserve explicit real labels."""
from .legacy import DATA
from .contracts import Reference
from evidence_cache import EvidenceCache
from real_episode_evidence import summarize_capture


def real_replay(path,cache_dir):
    path=path.resolve()
    if not path.is_relative_to(DATA/'complete_real'):
        raise ValueError('Select a stopped capture under complete_real')
    cache=EvidenceCache(cache_dir)
    try:
        result=summarize_capture(path,cache)
    finally:
        cache.close()
    value=result['value']
    event=value['recorded_stop_event']
    missing=[]
    if event and not (path/event['image']).is_file():
        missing.append('event RGB image missing')
    if any(c['status']!='ACCEPTED_BY_CURRENT_MAPPING' for c in value['camera_clocks'].values()):
        missing.append('timestamp mapping unavailable')
    hashes=result['cache']['dependency_sha256']
    execution=Reference('dataset',str((path/'execution_report.json').relative_to(DATA)),hashes['execution'])
    return dict(mode='stopped_real_capture',scope='RGB and encoder history',native_success=value['physical_task_success'],
                physical_label_source=value['physical_label_source'],recorded_stop=event,
                phase_end_width_mm={p['phase']:p['end_encoder_width_mm'] for p in value['phases']},
                source=execution.__dict__,event_alignment='UNKNOWN' if missing else 'recorded camera mapping',
                missing=missing,force_torque=None,depth=None,first_physical_divergence=None,
                root_cause_status='UNKNOWN',decision='UNKNOWN',hardware_commands=0,cache_hit=result['cache']['hit'])

"""Translate native probe output into offline typed events; gap alone is informational."""
from dataclasses import replace
from .first_divergence import event
from .fake_grasp import attachment_flags
from .legacy import DATA
from contact_evidence import inspect_state, restore
from episode_evidence import frame_at, locate


def contact_events(o,snapshot):
    if o.source_profile not in ('original','retained_final') or o.issue():
        return [event(o,'SENSOR_UNAVAILABLE')]
    result=[]
    for contact in snapshot['contacts']:
        distance=contact['distance_m']
        fact=replace(o,observed_value=distance)
        if not contact['solver_active']:
            result.append(event(fact,'CONTACT_INACTIVE',severity='info',status='OBSERVED',first_observed_time=o.timestamp_mono))
        if distance!=0:
            result.append(event(fact,'COLLISION_GAP',severity='info',status='OBSERVED',first_observed_time=o.timestamp_mono))
    return result


def archived_probe(ident,frame):
    import mujoco
    from .legacy import ROOT
    from .diagnosis import json_file,reference
    from .contracts import Observation
    path,episode,raw=locate(ident)
    stats=episode/'proprio_stats'
    row=frame_at(raw/'trajectory.jsonl',frame)
    extra=frame_at(stats/'privileged.jsonl',frame)
    if extra['frame']!=frame or extra['phase']!=row['phase']:
        raise ValueError('Frame mismatch')
    try:
        model=mujoco.MjModel.from_xml_path(str(stats/'model.xml'))
    except (ValueError,OSError) as exc:
        return dict(frame=frame,state_source='UNAVAILABLE',events=[],attachment_flags=None,
                    consistency='DEFER_UNKNOWN',reason=str(exc).splitlines()[0],model_ref=reference(stats/'model.xml',DATA).__dict__)
    data=mujoco.MjData(model)
    scope=restore(model,data,row,extra,stats/'initial_state.npz')
    snapshot=inspect_state(model,data,surfaces=False)
    r=json_file(path)
    refs=(reference(path,ROOT),*(reference(f,DATA) for f in [stats/'model.xml',stats/'privileged.jsonl',raw/'trajectory.jsonl',stats/'initial_state.npz']))
    fact=Observation(ident,r['actions'][0]['profile'],r['program_version']['source_sha256']['scripts/try_plug_pick_sim.py'],
                     r['initial_state']['task_config_sha256'],r['protocol_version'],'plug_insertion',r['initial_state']['target_index'],
                     row['phase'],str(frame),row['time'],'sim_time','verified','world','m','sim.privileged_contact','observed',0.,refs)
    flags=attachment_flags(model,data)
    from dataclasses import asdict
    return dict(frame=frame,state_source=scope,attachment_flags=flags,
                consistency='REJECT_AND_ROLLBACK' if flags else 'DEFER_UNKNOWN',
                events=[asdict(e) for e in contact_events(fact,snapshot)],direct_state_write_history='UNKNOWN',
                source='Simulation-only; restored state, historical external force and edits may be unobserved')

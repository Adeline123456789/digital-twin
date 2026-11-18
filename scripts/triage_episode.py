"""Local recorded-episode replay; no physics, publishing or hardware calls."""
import argparse
import json
from pathlib import Path
import sys
import time
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robot_agent.diagnosis import EventBuilder, ROOT, DATA, json_file
from robot_agent.policy_graph import replay_graph
from robot_agent.first_divergence import event
from robot_agent.supervisor import Supervisor
from diagnose_retention import diagnose_episode


def reconcile_stream(report,alarm):
    loss=next((e for e in report.events if e.kind=='GRASP_RETENTION_LOSS_OBSERVED'),None)
    def identity(e):
        return (e.kind,e.first_observed_frame,e.first_observed_time,e.confirmation_time) if e else None
    if identity(alarm)!=identity(loss):
        report.status='UNKNOWN'
        report.events.insert(0,event(alarm or loss,'STREAM_BATCH_DISAGREEMENT'))
        report.first_divergence=None


def replay(path, legacy=False, slow=None):
    started=time.perf_counter()
    if legacy:
        r=json_file(path)
        result=diagnose_episode(r)
        return dict(mode='legacy',episode=r['experiment_id'],diagnosis=result,
                    diagnostic_wall_seconds=time.perf_counter()-started,model_invoked=False,
                    input_tokens=0,output_tokens=0,simulation_wall_seconds=0)
    supervisor=Supervisor(slow,max_calls=int(slow is not None))
    latency=[]
    alarm=None
    def consume(fps,o):
        nonlocal alarm
        if not latency:
            supervisor.fast.max_interval=1.5/fps
        start=time.perf_counter_ns()
        decision=supervisor.fast.update(o)
        latency.append((time.perf_counter_ns()-start)/1e6)
        if alarm is None and decision.get('event'):
            alarm=decision['event']
    report=EventBuilder().build(path,consume)
    reconcile_stream(report,alarm)
    e=report.first_divergence
    context={}
    if e and e.kind=='TARGET_FRAME_RESIDUAL':
        r=json_file(path)
        # Archived playback binds the skill to this recorded source and configuration only.
        stats=DATA/'complete_sim/failed'/r['experiment_id']/'proprio_stats/privileged.jsonl'
        align=[row for row in map(json.loads,stats.read_text().splitlines()) if row['phase']=='align']
        context=dict(environment='simulation',target=e.target_index,source_target=e.target_index,
                     source_profile=e.source_profile,previous_source_profile=e.source_profile,frame=e.frame_id,units=e.units,
                     native_sha256=e.native_source_hash,previous_native_sha256=e.native_source_hash,
                     holder_shift_mm=r['initial_state']['holder_x_shift_mm'],previous_offset_mm=r['actions'][0]['offset_mm'][0],
                     align_contact_free=bool(align) and not any('task/strip' in (c['body1'],c['body2']) for c in align[-1]['contacts']),
                     retention_status=diagnose_episode(r)['status'],active_arms=r.get('active_arms',['left','right']),
                     world_model_version=e.world_model_version,policy_version=e.policy_version,residual=e.residual)
    decision=supervisor.diagnose(report,context)
    graph=replay_graph(e,context,supervisor.fast.router) if decision.get('plan') else None
    ordered=sorted(latency)
    from math import ceil
    return dict(mode='fastslow',episode=report.experiment_id,diagnosis=report.to_dict(),decision=decision,
                state_trace=supervisor.trace,graph=graph,stream_alarm=asdict(alarm) if alarm else None,
                measured_frames=len(latency),processing_ms={f'p{p}':ordered[ceil(len(ordered)*p/100)-1] if ordered else None for p in (50,95,99)},
                detection_delay_s=alarm.confirmation_time-alarm.first_observed_time if alarm and alarm.confirmation_time is not None else None,
                actual_planner_kind='local_slow' if supervisor.calls else 'compiled_alignment_v1' if graph else 'deterministic_triage',
                model_invoked=bool(supervisor.calls),input_tokens=supervisor.input_tokens if supervisor.usage_complete else None,
                output_tokens=supervisor.output_tokens if supervisor.usage_complete else None,
                planner_wall_seconds=supervisor.planner_seconds,simulation_wall_seconds=0,
                diagnostic_wall_seconds=time.perf_counter()-started,evidence_gate='REPLAY_ONLY')


def entry(episode,legacy=False,use_local_llm=False,contact_frame=None):
    path=Path(episode)
    if not path.is_file():
        if path.name!=episode:
            raise ValueError('Expected a repository result or an episode ID')
        path,=(ROOT/'experiments').glob('*/'+episode+'/result.json')
    from robot_agent.local_slow import local_diagnosis
    result=replay(path,legacy,local_diagnosis if use_local_llm else None)
    if contact_frame is not None:
        from robot_agent.contact_probe import archived_probe
        result['contact_probe']=archived_probe(result['episode'],contact_frame)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',nargs='?')
    p.add_argument('--stopped-real-capture',type=Path,help='Read a stopped real capture using the existing evidence cache')
    p.add_argument('--cache',type=Path,default=ROOT/'.local/grasp_six_rounds/cache')
    p.add_argument('--offline',action='store_true',help='Explicitly select recorded data (the only mode)')
    p.add_argument('--local-only',action='store_true',help='Results are always printed locally')
    p.add_argument('--contact-frame',type=int,help='Inspect one saved simulation contact frame')
    p.add_argument('--local-llm',action='store_true',help='Allow one bounded diagnosis via localhost:8080')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--legacy',action='store_true')
    mode.add_argument('--fastslow',action='store_true')
    a=p.parse_args()
    if bool(a.episode)==bool(a.stopped_real_capture):
        p.error('Choose a recorded simulation episode or --stopped-real-capture')
    if a.stopped_real_capture:
        from robot_agent.real_replay import real_replay
        print(json.dumps(real_replay(a.stopped_real_capture,a.cache),indent=2))
    else:
        entry(a.episode,a.legacy,a.local_llm,a.contact_frame)

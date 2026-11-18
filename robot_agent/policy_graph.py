"""Interpret the three trusted nodes. A historical replay can emit a request only."""
from pathlib import Path
from .diagnosis import json_file
from agent_loop import validate_plan


def replay_graph(event,context,router):
    graph=json_file(Path(__file__).with_name('policy_graph.json'))
    memory=router.retrieve_applicable(event,context)
    checks={'MATCH':memory['status']=='MATCH','VS002':bool(memory['plan']),
            'native_success_and_recording':False}
    steps=[]
    request=None
    for node in graph['nodes']:
        passed=checks[node['requires']]
        steps.append(dict(**node,passed=passed))
        if not passed:
            break
        if node['action']=='emit_sim_request':
            request=dict(adapter='simulation_replay_only',offset_mm=validate_plan(memory['plan']),
                         native_source_hash=event.native_source_hash,task_version=event.world_model_version)
    return dict(version=graph['version'],steps=steps,request=request,task_success=None)

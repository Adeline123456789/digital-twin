"""Evaluate the frozen diagnostic on local Episodes, including held-out success controls."""
import json
import time
from learn import ROOT, DATA, record, finish, save, digest
from diagnose_retention import diagnose_episode


def main():
    folder=ROOT/'experiments/research_stage4/round3'
    design=json.loads((folder/'design.json').read_text())
    out,result=record('research_stage4','Earliest grasp-retention diagnosis on frozen eight Episodes','offline_trace_analysis')
    start=time.monotonic();cases=[]
    try:
        for split in ['development_episodes','heldout_success_episodes']:
            for ident in design[split]:
                path=next((ROOT/'experiments').glob('*/'+ident+'/result.json'))
                source=json.loads(path.read_text())
                if split.startswith('heldout'):assert source['task_success'] and source['recording_verified']
                diagnosis=diagnose_episode(source)
                cases.append(dict(episode=ident,split=split,diagnosis=diagnosis,
                    original_task_success=source['task_success'],source_report=str(path.relative_to(ROOT)),
                    source_report_sha256=digest(path),trace_sha256=digest(DATA/'self_evolution'/ident/'trajectory.jsonl')))
        heldout=[x for x in cases if x['split'].startswith('heldout')]
        alarms=sum(x['diagnosis']['status']=='GRASP_RETENTION_LOSS_OBSERVED' for x in heldout)
        unknown=sum(x['diagnosis']['status']=='UNKNOWN' for x in heldout)
        positive=cases[0]['diagnosis']
        accepted=positive['status']=='GRASP_RETENTION_LOSS_OBSERVED' and positive['earliest_observed_phase']=='transport' and alarms==0 and unknown==0
        result.update(result='success' if accepted else 'failure',task_success=None,
            validation_status='SIMULATION_TRACE_DIAGNOSTIC_VALIDATED_LIMITED' if accepted else 'DIAGNOSTIC_GATE_FAILED',
            experiments=cases,heldout_false_alarms=alarms,heldout_unknowns=unknown,heldout_cases=len(heldout),
            independent_positive_validation_cases=0,accepted=accepted,
            program_version={'analyzer_sha256':digest(ROOT/'scripts/diagnose_retention.py'),'runner_sha256':digest(__file__)},
            tokens=0,planner_calls=0,physics_steps=0,new_episodes=0,research_orchestrator_tokens=None,human_interventions=None,
            privileged_information_used=True,privileged_fields_used=['object/gripper distance','simulator finger-object contact'],
            expected_effects=['Localize observed carry loss before downstream error; preserve all native outcomes'],
            lessons_learned=['A later IK error can be a consequence of a lost grasp; primitive order matters'],
            next_improvement='Use observed failure phase and applicable skills in the executable Agent entry, then compare complete collection cost')
    except Exception as exc:
        result['error']=str(exc);result['experiments']=cases
    finally:
        result['execution_time']=time.monotonic()-start;finish(out,result)
        save(folder/'result.json',{'experiment_id':result['experiment_id'],'result':result['result'],'accepted':result.get('accepted',False),'evidence':str((out/'result.json').relative_to(ROOT))})
    return out,result


if __name__=='__main__':main()

"""Collect the existing successful program from current legal entry, then freeze/replay it. No real motion."""
import argparse
import json
import os
from pathlib import Path
import sys
from benchmark_recipient_feedback import BatchBudget
from bounded_tools import baseline_case
from episode_evidence import locate
from learn import ROOT, DATA, digest, save
from trajectory_contract import freeze
from validate_shared_reference import validate


def build(connection,previous_run,output):
    previous=json.loads((previous_run/'ledger.json').read_text())
    connection_data=json.loads(connection.read_text())
    if not connection_data['passed'] or not connection_data['physical_sides_verified']:
        raise ValueError('Current right-arm read-only feedback required')
    arm=connection_data['arms'][0]
    if arm['interface']!='can_arm2' or any(m['enabled'] for m in arm['motors']):
        raise ValueError('Expected disabled physical right arm')
    # Reuse the existing folded-entry preparation; no controller is constructed.
    hardware_root=Path('/media/midea/F496AC8096AC4548/act_td3/piper_use_right')
    sys.path.insert(0,str(hardware_root))
    from replay_connector import prepare_entry, JOINT_BOUNDS
    entry=prepare_entry([*arm['joint_degrees'],.060/.095],JOINT_BOUNDS,True)[:6].tolist()
    source_path,_,raw=locate('20261116T125834Z_58df2069')
    source=json.loads(source_path.read_text())
    from episode_format import source_for
    source_for(source)
    os.environ.update(PIPER_ACTIVE_ARMS='right',PIPER_SIM_PROFILE='original',PIPER_TARGET_INDEX='3',PIPER_TASK_CONFIG=str(raw/'task.yaml'))
    output.mkdir(parents=True,exist_ok=False)
    budget=BatchBudget(previous['limits'],output/'ledger.json',previous,previous['started_at'])
    case=dict(baseline_case(source),name='Known successful program at current measured folded entry',start_right_deg=entry,shift_mm=0)
    save(output/'request.json',dict(control='Existing successful program, no new model decision or teacher-transfer claim',
        source_episode=source['experiment_id'],connection_sha256=digest(connection),measured_joint_degrees=arm['joint_degrees'],
        native_folded_entry_joint_degrees=entry,case=case,model_tokens_before=budget.tokens(),
        successful_simulation_requires_real_pair=True))
    result=budget.simulate(case,'Current-entry paired control')
    save(output/'collection_result.json',result)
    if not result['task_success']:return result
    _,episode,trial=locate(result['experiment_id'])
    reference=DATA/'self_evolution'/'recipient_feedback_pairs'/result['experiment_id']/'reference'
    frozen=freeze(episode,trial,reference)
    validation=validate(trial,reference)
    pair=dict(simulation_episode=result['experiment_id'],reference_dataset_path=str(reference.relative_to(DATA)),
        reference_id=frozen['trajectory_id'],reference_validation=validation,
        real_status='PENDING' if validation['task_success'] and validation['all_substeps_valid'] else 'REFERENCE_VALIDATION_FAILED',
        real_episode=None,source='deterministic known-program control; not recipient discovery',privileged_information_used=True)
    save(output/'pair.json',pair)
    print(json.dumps(dict(episode=result['experiment_id'],real_status=pair['real_status'],reference=str(reference))),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--connection',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    build(a.connection,a.previous_run,a.output)

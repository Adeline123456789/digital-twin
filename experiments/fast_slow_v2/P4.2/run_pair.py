"""Two new full collections, same native settings. The experimental factor is offline triage."""
import json
import os
import sys
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[2]
sys.path.insert(0,str(root/'scripts'))
from learn import SIM,digest
from benchmark_collection import run_case
from triage_episode import replay


def main():
    design=json.loads((out/'pair_design.json').read_text())
    if digest(SIM/'scripts/try_plug_pick_sim.py')!=design['current_native']:
        raise ValueError('Native version changed')
    os.environ.update(PIPER_ACTIVE_ARMS='dual',PIPER_SIM_PROFILE='original',PIPER_TARGET_INDEX='3')
    if os.environ.get('PIPER_TASK_CONFIG'):
        raise ValueError('Unexpected alternate task')
    for mode in design['order']:
        # Exclusive slot is consumed even when the collector fails.
        with (out/(mode+'_start.json')).open('x') as f:
            json.dump(dict(mode=mode,scope='one complete simulation attempt'),f)
        r=run_case(dict(name=mode,offset_mm=0.,shift_mm=0.),True,label='V2 matched full simulation',kind='fast_slow_collection',publish=False)
        ident=r['experiment_id']
        summary={k:r.get(k) for k in ('experiment_id','result','task_success','recording_verified',
                                      'recording_report','execution_time','simulation_seconds','collection_component_seconds','recording_error','error')}
        summary['source_hash']=r.get('program_version',{}).get('source_sha256',{}).get('scripts/try_plug_pick_sim.py')
        summary['task_sha256']=r.get('initial_state',{}).get('task_config_sha256')
        if r.get('recording_verified'):
            result=replay(root/f'experiments/fast_slow_collection/{ident}/result.json',legacy=mode=='legacy')
            summary['triage_status']=result['diagnosis']['status']
            summary['triage_metrics']={k:result[k] for k in ('diagnostic_wall_seconds','model_invoked','input_tokens','output_tokens','processing_ms') if k in result}
        with (out/(mode+'_result.json')).open('x') as f:
            json.dump(summary,f,indent=2)
        print(json.dumps(summary),flush=True)
        if not r.get('recording_verified'):
            break


if __name__=='__main__':
    main()

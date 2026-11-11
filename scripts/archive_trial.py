"""Route newly produced trial evidence without touching existing episodes or uploading raw data."""
import argparse
import json
from pathlib import Path
import shutil
from learn import DATA,ROOT,save,digest
SUCCESS=DATA/'complete_sim/succeed/20261111_sim_calibrated_pick_insert_000001'
def main():
    p=argparse.ArgumentParser();p.add_argument('experiment_id');a=p.parse_args()
    matches=list((ROOT/'experiments').glob('*/'+a.experiment_id+'/result.json'))
    if len(matches)!=1:raise ValueError('Expected one experiment record')
    report=json.loads(matches[0].read_text())
    if report['environment']!='simulation' or report['task_success'] is None:raise ValueError('Only observed simulation results')
    source=DATA/'self_evolution'/a.experiment_id
    if report['task_success']:
        dest=SUCCESS/'proprio_stats'/a.experiment_id/'source_trial'
    else:dest=DATA/'complete_sim/failed'/a.experiment_id/'source_trial'
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,dest) # fails if already present; no overwrite
    index=dict(experiment_id=a.experiment_id,success=report['task_success'],
               source_trial=str(dest.relative_to(DATA)),videos_status='NOT_RECORDED_BY_HEADLESS_TRIAL',
               force_sensor_present=False,files={f.name:digest(f) for f in dest.iterdir() if f.is_file()})
    save(matches[0].parent/'dataset_index.json',index)
    save(dest.parent/'outcome.json',index)
    print(dest)
if __name__=='__main__':main()

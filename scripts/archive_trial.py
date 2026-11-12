"""Route newly produced trial evidence without touching existing episodes or uploading raw data."""
import argparse
import json
from pathlib import Path
import shutil
import uuid
from learn import DATA,ROOT,save,digest
def main():
    p=argparse.ArgumentParser();p.add_argument('experiment_id');a=p.parse_args()
    matches=list((ROOT/'experiments').glob('*/'+a.experiment_id+'/result.json'))
    if len(matches)!=1:raise ValueError('Expected one experiment record')
    report=json.loads(matches[0].read_text())
    if report['environment']!='simulation' or report['task_success'] is None:raise ValueError('Only observed simulation results')
    source=DATA/'self_evolution'/a.experiment_id
    if report['task_success']:
        dest=DATA/'self_evolution/archived_trials'/a.experiment_id/'source_trial'
    else:
        from episode_format import source_for, convert_group
        dest=DATA/'complete_sim/failed'/a.experiment_id
        if dest.exists():raise FileExistsError('Episode already exists; do not overwrite')
        work=DATA/'self_evolution'/('format_'+a.experiment_id+'_'+uuid.uuid4().hex[:8]);work.mkdir()
        result=convert_group([dict(id=a.experiment_id,source=str(source_for(report)),trial=str(source),
                                  trajectory_sha256=digest(source/'trajectory.jsonl'))],work)[0]
        save(matches[0].parent/'dataset_index.json',dict(experiment_id=a.experiment_id,
            archive_status='converted',dataset_relative_path=str(dest.relative_to(DATA)),verification=result['verification']))
        work.rmdir()
        print(dest)
        return
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,dest) # fails if already present; no overwrite
    index=dict(experiment_id=a.experiment_id,success=report['task_success'],
               source_trial=str(dest.relative_to(DATA)),videos_status='NOT_RECORDED_BY_HEADLESS_TRIAL',
               force_sensor_present=False,files={f.name:digest(f) for f in dest.iterdir() if f.is_file()})
    save(matches[0].parent/'dataset_index.json',index)
    save(dest.parent/'outcome.json',index)
    print(dest)
if __name__=='__main__':main()

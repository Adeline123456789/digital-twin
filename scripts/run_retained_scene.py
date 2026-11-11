"""Execute the retained final scene's documented command without altering its source/config."""
import os
import json
import subprocess
import time
import sys
from learn import DATA,ROOT,record,finish,digest,save

def main():
    source=DATA/'calibration/offline_workspace_20261110'
    out,r=record('evaluation','Retained final scene documented collection command','simulation')
    raw=DATA/'self_evolution'/r['experiment_id'];start=time.monotonic()
    r['program_version']={'source_root_dataset_relative':str(source.relative_to(DATA)),'git_head':None,
        'source_sha256':{str(p.relative_to(source)):digest(p) for d in ('scripts','src','configs') for p in (source/d).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json')},'adapter_sha256':digest(__file__)}
    args=['--task','configs/tasks/real_connector_final.yaml','--output',str(raw),'--speed','25','--smooth-timing','--grasp-height','.012','--grasp-roll-deg','30','--align-corrections','4','--grasp-opening','0','--start-right-deg','0','0','0','0','0','0']
    r['initial_state']={'scene':'retained real_connector_final.yaml','right_joints_deg':[0]*6,'physics_changes':False}
    r['actions']=[{'script':'scripts/try_plug_pick_sim.py','arguments':[x.replace(str(DATA),'$DATA') for x in args]}]
    local=ROOT/'.local'/r['experiment_id'];local.mkdir(parents=True)
    try:
        with (local/'stdout.log').open('w') as log:
            p=subprocess.run([str(source/'.venv-mujoco/bin/python'),'scripts/try_plug_pick_sim.py',*args],cwd=source,env=dict(os.environ,MUJOCO_GL='egl'),stdout=log,stderr=subprocess.STDOUT,timeout=900)
        report=json.loads((raw/'report.json').read_text())
        success=bool(p.returncode==0 and report['complete_pick_insert'] and report['max_penetration_m']<=.0002)
        r.update(result='success' if success else 'failure',task_success=success,observed_effects=[report],
                 validation_status='SIMULATION_VERIFIED' if success else 'SIMULATION_FAILED',
                 simulation_seconds=report['frames']/30,failure_signature=report.get('error'),
                 lessons_learned=['Retained final source exists despite deleted intermediate sysid trials; results must be version-specific.'],
                 next_improvement='Use this retained final source for subsequent Agent integration; do not pool with older source trials')
        r['evidence']=[{'dataset_relative_path':str(f.relative_to(DATA)),'sha256':digest(f)} for f in raw.iterdir() if f.is_file()]
    except Exception as exc:r['error']=str(exc).replace(str(DATA),'$DATA')
    finally:r['execution_time']=time.monotonic()-start;finish(out,r)
    if r['task_success'] is not None:subprocess.run([sys.executable,str(ROOT/'scripts/archive_trial.py'),r['experiment_id']],check=True)
if __name__=='__main__':main()

"""Replay each selected historical real command timeline using its retained initial state."""
import json
import os
import subprocess
import time
import sys
from learn import ROOT,DATA,SIM,record,finish,digest

def run(name):
    out,r=record('diagnosis','Physical replay of real failure '+name,'simulation')
    capture=DATA/'complete_real/failed'/name/'raw/task'
    start=time.monotonic();raw=DATA/'self_evolution'/r['experiment_id']
    local=ROOT/'.local'/r['experiment_id'];local.mkdir(parents=True)
    try:
        real=json.loads((capture/'execution_report.json').read_text())
        from pathlib import Path
        trial=Path(real['source_trial'])
        r['initial_state']={'source_episode':name,'trial_dataset_relative':str(trial.relative_to(DATA)),
            'initial_state_sha256':digest(trial/'initial_state.npz'),'task_config_sha256':digest(trial/'task.yaml')}
        r['actions']=[{'command_csv_sha256':digest(capture/'commanded_probe.csv'),'source':'recorded real targets, resampled to physics time'}]
        cmd=[str(SIM/'.venv-mujoco-hardware/bin/python'),str(SIM/'scripts/replay_plug_real_in_sim.py'),
             '--capture',str(capture),'--trial',str(trial),'--output',str(raw)]
        with (local/'stdout.log').open('w') as log:
            proc=subprocess.run(cmd,cwd=SIM,env=dict(os.environ,MUJOCO_GL='egl'),stdout=log,stderr=subprocess.STDOUT,timeout=900)
        report=json.loads((raw/'report.json').read_text())
        if 'trajectory_provenance' in report:report['trajectory_provenance']=str((capture/'commanded_probe.csv').relative_to(DATA))
        success=bool(proc.returncode==0 and report['complete_pick_insert'] and report['all_substeps_valid'])
        r.update(result='success' if success else 'failure',task_success=success,
            validation_status='SIMULATION_VERIFIED' if success else 'SIMULATION_FAILED',observed_effects=[report],
            real_task_success=False,simulation_seconds=report['frames']/30,
            lessons_learned=['Recorded real targets were physically replayed; simulated object motion is not measured real object ground truth.'],
            next_improvement='Compare onset and gripper/pose evidence; do not infer a unique physical cause from matching final failure')
        r['evidence']=[{'dataset_relative_path':str(f.relative_to(DATA)),'sha256':digest(f)} for f in raw.iterdir() if f.is_file()]
        r['limitations']=['Uses historical initial-state snapshot and original retained runtime, not newer final scene',
                          'Requested target timestamps resampled at 30 Hz; not exact CAN transmit times',
                          'Same final failure is insufficient evidence of the same causal mechanism']
    except Exception as exc:r['error']=str(exc).replace(str(DATA),'$DATA')
    finally:r['execution_time']=time.monotonic()-start;finish(out,r)
    if r['task_success'] is not None:subprocess.run([sys.executable,str(ROOT/'scripts/archive_trial.py'),r['experiment_id']],check=True)
if __name__=='__main__':run(sys.argv[1])

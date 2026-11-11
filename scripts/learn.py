"""Additive, offline-only adapter for the existing Piper MuJoCo trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SIM = Path(os.environ.get('PIPER_SIM_ROOT', '/media/midea/F496AC8096AC4548/piper_simulation'))
DATA = Path(os.environ.get('PIPER_DATA_ROOT', '/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def version():
    files = [p for folder in ('scripts','src','configs') for p in (SIM/folder).rglob('*')
             if p.is_file() and p.suffix in ('.py','.yaml','.json')]
    return dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=SIM,text=True).strip(),
                source_sha256={str(p.relative_to(SIM)):digest(p) for p in sorted(files)},
                adapter_sha256=digest(__file__), dirty=True)

def record(kind, task, environment):
    ident=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'experiments'/kind/ident
    out.mkdir(parents=True, exist_ok=False)
    return out,dict(experiment_id=ident,timestamp=datetime.now(timezone.utc).isoformat(),
        model='deterministic adapter; no LLM inference',task=task,environment=environment,
        program_version=version(),protocol_version=digest(ROOT/'protocol/robot_problem_solving.md'),
        memory_version={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'memory').rglob('*.json')},
        initial_state={},actions=[],expected_effects=['plug lifted, inserted, released and stably seated'],
        observed_effects=[],failure_signature=None,root_cause='UNKNOWN',root_cause_status='unknown',
        hypotheses=[],experiments=[],repair={},result='infrastructure_error',task_success=None,
        tokens=None,llm_inference_seconds=None,execution_time=None,robot_execution_seconds=0,
        human_interventions=0,lessons_learned=[],next_improvement='Resolve missing evidence',
        validation_status='UNVERIFIED')

def finish(out,r):
    save(out/'result.json',r)
    (out/'REPORT.md').write_text('# '+r['experiment_id']+'\n\n'+
        '\n'.join(f'- **{k}**: {json.dumps(r[k],ensure_ascii=False)}' for k in
        ('task','environment','result','task_success','validation_status','failure_signature','root_cause',
         'root_cause_status','execution_time','tokens','lessons_learned','next_improvement'))+'\n\nSee result.json for provenance and evidence hashes. Raw data stays in the existing dataset.\n')
    print(out.relative_to(ROOT),r['result'],flush=True)

def retrieve(signature, limit=5):
    matches=[]
    for p in sorted((ROOT/'memory').rglob('*.json')):
        item=json.loads(p.read_text())
        if signature in item.get('signatures',[]): matches.append(item)
    return matches[:limit]

def decide(signature, repair, version_name='v2'):
    if version_name=='v1': return 'ALLOW_SIMULATION'
    for item in retrieve(signature):
        if repair in item.get('ineffective_repairs',[]): return 'REJECT_REPEAT_REQUIRE_NEW_EVIDENCE'
    return 'UNKNOWN_TASK_MODE' if not retrieve(signature) else 'ALLOW_SIMULATION'

def simulate(kind, offset, shift, label):
    out,r=record(kind,label,'simulation')
    raw=DATA/'self_evolution'/r['experiment_id'];raw.parent.mkdir(parents=True, exist_ok=True)
    start=time.monotonic()
    try:
        # YAML is already installed in the existing MuJoCo environment.
        import yaml
        cfg=yaml.safe_load((SIM/'configs/tasks/real_connector_calibrated.yaml').read_text())
        cfg['holder_xy_m'][0]+=shift/1000
        local=ROOT/'.local'/r['experiment_id'];local.mkdir(parents=True)
        task=local/'task.yaml';task.write_text(yaml.safe_dump(cfg))
        cmd=[str(SIM/'.venv-mujoco-hardware/bin/python'),str(SIM/'scripts/try_plug_pick_sim.py'),
             '--output',str(raw),'--task',str(task),'--smooth-timing','--speed','12',
             '--grasp-height','.020','--insertion-offset-mm',str(offset),'0']
        r['initial_state']={'holder_x_shift_mm':shift,'task_config_sha256':digest(task)}
        r['actions']=[{'script':'try_plug_pick_sim.py','grasp_height_m':.020,'offset_mm':[offset,0],
                       'speed_deg_s':12,'smooth_timing':True}]
        r['experiments']=[{'changed_variable':'insertion_offset_mm','value':offset,'heldout_initial_shift_mm':shift}]
        env=dict(os.environ,MUJOCO_GL='egl')
        with (local/'stdout.log').open('w') as log:
            p=subprocess.run(cmd,cwd=SIM,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        r['process_returncode']=p.returncode
        if not (raw/'report.json').exists(): raise RuntimeError('Simulation produced no report; inspect local stdout')
        report=json.loads((raw/'report.json').read_text())
        r['observed_effects']=[report]
        r['task_success']=bool(report['complete_pick_insert'] and report['max_penetration_m']<=.0002 and p.returncode==0)
        r['result']='success' if r['task_success'] else 'failure'
        r['validation_status']='SIMULATION_VERIFIED' if r['task_success'] else 'SIMULATION_FAILED'
        r['failure_signature']=None if r['task_success'] else report.get('error','UNKNOWN')
        r['simulation_seconds']=report['frames']/30
        r['lessons_learned']=['Outcome applies only to this simulation configuration; real cause remains UNKNOWN.']
        r['next_improvement']='Compare matched initial states and independent holdout shifts before accepting repair'
        # Publish only sanitized numerical reports; raw images and trajectories stay outside Git.
        r['evidence']=[{'dataset_relative_path':str(f.relative_to(DATA)), 'sha256':digest(f)}
                       for f in raw.iterdir() if f.is_file()]
    except Exception as e:
        r['error_type']=type(e).__name__
        r['error']=str(e).replace(str(SIM),'$PIPER_SIM_ROOT').replace(str(DATA),'$PIPER_DATA_ROOT').replace(str(ROOT),'$PROJECT_ROOT')
    finally:
        r['execution_time']=time.monotonic()-start
        finish(out,r)
    return out,r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',default='improvement',choices=['baseline','improvement','evaluation'])
    p.add_argument('--offset-mm',type=float,default=0);p.add_argument('--holder-shift-mm',type=float,default=0)
    p.add_argument('--label',default='yellow plug insertion')
    p.add_argument('--retrieve');p.add_argument('--repair',default='');a=p.parse_args()
    if a.retrieve: print(json.dumps(dict(memories=retrieve(a.retrieve),decision=decide(a.retrieve,a.repair)),ensure_ascii=False,indent=2))
    else: simulate(a.kind,a.offset_mm,a.holder_shift_mm,a.label)

"""32-frame diagnostic from a frozen native trial; output directory must be local, new and unique."""
import sys,os,json,time,hashlib
from pathlib import Path
import numpy as np,yaml,mujoco,av
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from learn import SIM, DATA
sys.path.insert(0,str(SIM/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.contracts import Command
from piper_sim.mixed_recording import configure_rigid_collection,checked_physics_step
trial=DATA/'self_evolution/20261112T073436Z_43d3e2c2'
out=Path(sys.argv[1]);out.mkdir()
sim=MujocoBackend(dict(load(),task_scene=yaml.safe_load((trial/'task.yaml').read_text())),cameras=True)
configure_rigid_collection(sim)
with np.load(trial/'initial_state.npz') as z:
 sim.data.qpos[:]=z['qpos'];sim.data.qvel[:]=z['qvel'];sim.data.ctrl[:]=z['ctrl']
mujoco.mj_forward(sim.model,sim.data)
rows=[json.loads(l) for l in (trial/'trajectory.jsonl').read_text().splitlines()][:32]
frames={k:[] for k in ('overhead','left_wrist','right_wrist')}
for i,row in enumerate(rows):
 sim.apply(Command(row['action'],sim.sim_time,i));checked_physics_step(sim,.0002)
 obs=sim.read()
 for key in frames:frames[key].append(obs.images[key])
for key,v in frames.items():np.save(out/(key+'.npy'),np.stack(v))
sim.close()
print(out,flush=True)

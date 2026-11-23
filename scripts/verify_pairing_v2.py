"""Reset-only audit: paired policies start from identical integration/model state."""
import hashlib
import json
from pathlib import Path
import sys
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent.residual_env import LocalResidualEnv, ROOT
from evaluate_residual_v2 import cases


def state_hash(env):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.sim.model, spec))
    mujoco.mj_getState(env.sim.model, env.sim.data, state, spec)
    return hashlib.sha256(state.tobytes() + env.sim.model.body_pos.tobytes()).hexdigest()


def main():
    folder = ROOT / 'experiments/event_residual_mvp_v2'
    jobs = cases(json.loads((folder/'design.json').read_text())) + cases(json.loads((folder/'phase2_design.json').read_text()))
    histories = {}
    for name, frames, visual in [('nominal', 15, False), ('phase_policy', 1, False), ('history_policy', 15, True)]:
        env = LocalResidualEnv(training=False, history_frames=frames, visual_features=visual, visual_dropout=0.)
        histories[name] = {}
        for case in jobs:
            env.reset(seed=case['seed'], options={'perturbation':case['perturbation']})
            histories[name][case['id']] = state_hash(env)
        env.close()
    assert histories['nominal'] == histories['phase_policy'] == histories['history_policy']
    result = dict(status='PASS', conditions=len(jobs), physics_steps=0, identity='Full MuJoCo integration state + model body positions',
        modes=list(histories), initial_state_hashes=histories['nominal'],
        scope='Reset equivalence of frozen local environment across policy history/RGB modes; same seeds and explicit perturbations used by paired rollouts')
    (folder/'initial_state_pairing.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PAIRING_PASS',len(jobs),flush=True)


if __name__ == '__main__':
    main()

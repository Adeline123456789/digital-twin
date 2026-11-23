"""Seeded compute witness and actual local frozen replay; no training or hardware."""
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent.residual_env import LocalResidualEnv, DATA


def main():
    torch.manual_seed(0)
    x = torch.randn(8, 8, device='cuda')
    print('KERNEL_WITNESS', (x @ x).shape, torch.cuda.get_device_name(), float((x @ x).sum()), flush=True)
    start = time.monotonic()
    env = LocalResidualEnv(training=False)
    stats = DATA / 'complete_sim/succeed/20261123T090549Z_a35ab76b/proprio_stats/privileged.jsonl'
    # Evaluator-only reference states; never supplied to the Student observation.
    with stats.open() as stream:
        expected = [json.loads(line)['physics_state'] for line in stream]
    obs, _ = env.reset(seed=0)
    max_error = 0.
    while True:
        obs, _, done, _, info = env.step(np.zeros(2))
        snap = expected[env.index]
        state = np.empty(mujoco.mj_stateSize(env.sim.model, snap['spec']))
        mujoco.mj_getState(env.sim.model, env.sim.data, state, snap['spec'])
        max_error = max(max_error, float(np.max(abs(state - snap['values']))))
        assert obs.shape == (480,) and np.isfinite(obs).all()
        if done:
            break
    env.close()
    assert info['is_success'] and not info['collision']
    assert max_error < 1e-8, max_error
    result = dict(**info, max_integration_state_error=max_error, wall_s=time.monotonic() - start,
                  sample_scope='every 2 native frames (15 Hz policy); original full replay separately validated')
    output = Path('.local/event_residual_mvp_v2/runtime_witness.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('LOCAL_REPLAY_WITNESS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

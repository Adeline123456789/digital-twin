"""Predeclared paired local physics pilot, outside formal Episode roots."""
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent.residual_env import LocalResidualEnv, DATA, ROOT


def search_action(step):
    angle = step / 15 * 2 * np.pi / 2
    radius = min(.75, step / 30 * .75)
    return radius * np.array([np.cos(angle), np.sin(angle)])


def initialize():
    global ENV
    ENV = LocalResidualEnv(training=False)


def rollout(item):
    case, policy = item
    obs, _ = ENV.reset(seed=case['seed'], options={'perturbation': case['perturbation']})
    reward_sum, actions = 0., []
    start = time.monotonic()
    for step in range(100):
        action = np.zeros(2) if policy == 'B0' else search_action(step)
        obs, reward, done, _, info = ENV.step(action)
        reward_sum += reward
        actions.append(action.tolist())
        if done:
            result = dict(case_id=case['id'], policy=policy, seed=case['seed'], **info,
                          reward_sum=reward_sum, wall_s=time.monotonic() - start)
            raw = DATA / 'self_evolution/event_residual_mvp_v2/pilot'
            raw.mkdir(parents=True, exist_ok=True)
            (raw / f"{case['id']}_{policy}.json").write_text(json.dumps(dict(**result, actions=actions), indent=2) + '\n')
            return result
    raise RuntimeError('Frozen local horizon did not terminate')


def main():
    design = json.loads((ROOT / 'experiments/event_residual_mvp_v2/design.json').read_text())
    jobs = [(case, policy) for case in design['pilot_cases'] for policy in ('B0', 'B1')]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=initialize,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(rollout, jobs):
            results.append(result)
            print(result['case_id'], result['policy'], result['is_success'], result['collision'], flush=True)
    path = ROOT / 'experiments/event_residual_mvp_v2/pilot_results.json'
    path.write_text(json.dumps(dict(scope='Simulation-only paired pilot; not formal held-out results',
                                   cases=results), indent=2) + '\n')


if __name__ == '__main__':
    main()

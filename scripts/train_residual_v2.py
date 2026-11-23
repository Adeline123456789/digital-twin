"""Compact PPO on deployable observations; only residual weights are optimized."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent.residual_env import LocalResidualEnv, DATA, ROOT
from learn import digest


def make_env(history, visual, reward, stage):
    return Monitor(LocalResidualEnv(history_frames=history, visual_features=visual, reward_kind=reward,
                                    perturbation_stage=stage))


class Progress(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output, self.episodes = output, []
        self.started = time.monotonic()

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                row = {**info, 'training_steps': self.num_timesteps}
                row.pop('terminal_observation', None)
                self.episodes.append(row)
        if self.n_calls % 128 == 0:
            recent = self.episodes[-32:]
            status = dict(steps=self.num_timesteps, episodes=len(self.episodes),
                recent_success=sum(x['is_success'] for x in recent) / max(1, len(recent)),
                wall_s=time.monotonic() - self.started)
            self.output.write_text(json.dumps(status, indent=2) + '\n')
            print('TRAINING_PROGRESS', json.dumps(status), flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['B2_structured', 'B3_structured', 'B3_sparse', 'B3_actuator'], required=True)
    args = parser.parse_args()
    design_name = 'phase2_design.json' if args.variant == 'B3_actuator' else 'design.json'
    design = json.loads((ROOT / 'experiments/event_residual_mvp_v2' / design_name).read_text())
    config = design['training'][args.variant]
    torch.set_num_threads(1)
    output = DATA / 'self_evolution/event_residual_mvp_v2/training' / args.variant
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in
        [Path(__file__), ROOT / 'robot_agent/residual_env.py', ROOT / 'robot_agent/residual_controller.py',
         ROOT / 'robot_agent/proprio_history.py', ROOT / 'robot_agent/observation_adapter.py']}
    snapshots = output / 'sources'
    for relative in source_hashes:
        copy = snapshots / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((ROOT / relative).read_bytes())
    (output / 'launch.json').write_text(json.dumps(dict(config=config, source_hashes=source_hashes), indent=2) + '\n')
    factory = partial(make_env, config['history_frames'], config['visual_features'], config['reward'],
                      config.get('stage', 'geometry'))
    env = SubprocVecEnv([factory] * 8, start_method='spawn')
    env.seed(config['seed'])
    if args.variant == 'B3_actuator':
        parent = json.loads((ROOT / 'experiments/event_residual_mvp_v2/training_B3_structured.json').read_text())
        if digest(DATA / parent['policy']['dataset_relative_path']) != parent['policy']['sha256']:
            raise ValueError('Parent residual checkpoint changed')
        policy = PPO.load(DATA / parent['policy']['dataset_relative_path'], env=env, device='cpu', seed=config['seed'])
    else:
        policy = PPO('MlpPolicy', env, seed=config['seed'], device='cpu', n_steps=128, batch_size=256,
            n_epochs=5, learning_rate=.0003, gamma=.99, gae_lambda=.95, ent_coef=.001,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])), verbose=0)
    callback = Progress(output / 'progress.json')
    start = time.monotonic()
    policy.learn(total_timesteps=config['steps'], callback=callback)
    policy.save(str(output / 'policy'))
    env.close()
    (output / 'episodes.json').write_text(json.dumps(callback.episodes, indent=2) + '\n')
    result = dict(variant=args.variant, config=config, actual_steps=policy.num_timesteps,
        episodes=len(callback.episodes), wall_s=time.monotonic()-start, source_hashes=source_hashes,
        policy=dict(dataset_relative_path=str((output / 'policy.zip').relative_to(DATA)), sha256=digest(output / 'policy.zip')),
        observation_scope='right motor/command history; optional RGB candidates with dropout, no privileged channels',
        scope='Simulation-only; single training seed per ablation; no real transfer inference')
    (ROOT / f'experiments/event_residual_mvp_v2/training_{args.variant}.json').write_text(json.dumps(result, indent=2) + '\n')
    print('TRAINING_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

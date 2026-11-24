"""Train one fixed-budget P0 History PPO seed on CONTACT_MEDIATED_DRIFT_V1."""
import argparse
from datetime import datetime, timedelta, timezone
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from learn import DATA, digest
from robot_agent.full_task_residual_env import FullTaskContactDriftEnv


def make_env(design_path):
    return Monitor(FullTaskContactDriftEnv(design_path=design_path, training=True))


class Progress(BaseCallback):
    def __init__(self, output, total_steps):
        super().__init__()
        self.output = output
        self.total_steps = total_steps
        self.episodes = []
        self.started = time.monotonic()

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                row = {**info, 'training_steps': self.num_timesteps}
                row.pop('terminal_observation', None)
                self.episodes.append(row)
        if self.n_calls % 64 == 0:
            elapsed = time.monotonic() - self.started
            remaining = elapsed * max(0, self.total_steps - self.num_timesteps) / max(1, self.num_timesteps)
            recent = self.episodes[-24:]
            status = {
                'steps': self.num_timesteps,
                'total_steps': self.total_steps,
                'episodes': len(self.episodes),
                'recent_success_rate': (
                    sum(row['is_success'] for row in recent) / len(recent) if recent else None),
                'wall_s': elapsed,
                'eta_s': remaining,
                'estimated_completion_utc': (
                    datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat(),
            }
            self.output.write_text(json.dumps(status, indent=2) + '\n')
            print('TRAINING_PROGRESS', json.dumps(status), flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    design_path = ROOT / 'experiments/residual_fulltask_mvp/contact_drift_design.json'
    design = json.loads(design_path.read_text())
    training = design['training']
    if args.seed not in training['seeds']:
        parser.error('--seed must be frozen in contact_drift_design.json')
    if args.workers != training['parallel_envs_per_seed']:
        parser.error('--workers must match the frozen parallel_envs_per_seed')

    output = DATA / 'self_evolution/residual_fulltask_mvp/contact_drift/training' / f'seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=False)
    result_path = ROOT / 'experiments/residual_fulltask_mvp' / f'contact_drift_training_seed_{args.seed}.json'
    source_paths = [
        Path(__file__),
        ROOT / 'robot_agent/full_task_residual_env.py',
        ROOT / 'robot_agent/full_task_observation.py',
        ROOT / 'robot_agent/residual_controller.py',
        ROOT / 'robot_agent/nominal_reference.py',
    ]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    snapshots = output / 'sources'
    for relative in source_hashes:
        destination = snapshots / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    launch = {
        'seed': args.seed,
        'workers': args.workers,
        'training': training,
        'design_sha256': digest(design_path),
        'source_hashes': source_hashes,
        'started_utc': datetime.now(timezone.utc).isoformat(),
    }
    (output / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')

    torch.set_num_threads(1)
    factory = partial(make_env, design_path)
    env = SubprocVecEnv([factory] * args.workers, start_method='spawn')
    env.seed(args.seed)
    policy = PPO(
        'MlpPolicy', env, seed=args.seed, device='cpu',
        n_steps=training['n_steps'], batch_size=training['batch_size'],
        n_epochs=training['n_epochs'], learning_rate=training['learning_rate'],
        gamma=training['gamma'], gae_lambda=training['gae_lambda'],
        ent_coef=training['ent_coef'],
        policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]}},
        verbose=0,
    )
    callback = Progress(output / 'progress.json', training['steps_per_seed'])
    started = time.monotonic()
    policy.learn(total_timesteps=training['steps_per_seed'], callback=callback)
    policy.save(str(output / 'policy'))
    env.close()
    wall_s = time.monotonic() - started
    (output / 'episodes.json').write_text(json.dumps(callback.episodes, indent=2) + '\n')
    result = {
        'schema_version': 1,
        'benchmark': 'CONTACT_MEDIATED_DRIFT_V1',
        'variant': 'B2_HISTORY_PPO_STRUCTURED',
        'seed': args.seed,
        'config': training,
        'actual_steps': policy.num_timesteps,
        'episodes': len(callback.episodes),
        'wall_s': wall_s,
        'source_hashes': source_hashes,
        'design_sha256': digest(design_path),
        'policy': {
            'dataset_relative_path': str((output / 'policy.zip').relative_to(DATA)),
            'sha256': digest(output / 'policy.zip'),
        },
        'observation_scope': 'P0 15-frame right proprioception/nominal/feedback/gripper/phase history',
        'privileged_student_inputs': 0,
        'hardware_executions': 0,
        'real_data_reads': 0,
        'scope': 'Simulation-only fixed-budget final checkpoint; no held-out checkpoint selection',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
    }
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print('TRAINING_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

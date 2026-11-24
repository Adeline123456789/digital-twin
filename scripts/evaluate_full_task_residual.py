"""Evaluate one deployable baseline on the frozen CONTACT_MEDIATED_DRIFT_V1 held-out set."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from learn import DATA, digest
from pilot_residual_v2 import search_action
from robot_agent.full_task_residual_env import FullTaskContactDriftEnv, sample_contact_drift


def heldout_conditions(design):
    rng = np.random.default_rng(design['heldout']['generator_seed'])
    return [
        {
            **sample_contact_drift(
                design['distribution'], rng, f'heldout_{index:03d}'),
            'seed': design['heldout']['condition_seed_start'] + index,
        }
        for index in range(design['heldout']['conditions'])
    ]


def initialize(variant, policy_seed, design_path, output_root):
    global ENV, POLICY, VARIANT, POLICY_SEED, OUTPUT_ROOT
    import torch

    torch.set_num_threads(1)
    VARIANT = variant
    POLICY_SEED = policy_seed
    OUTPUT_ROOT = Path(output_root)
    ENV = FullTaskContactDriftEnv(design_path=design_path, training=False)
    POLICY = None
    if variant == 'B2':
        from stable_baselines3 import PPO

        record_path = ROOT / 'experiments/residual_fulltask_mvp' / f'contact_drift_training_seed_{policy_seed}.json'
        record = json.loads(record_path.read_text())
        if record['design_sha256'] != digest(Path(design_path)):
            raise ValueError('Policy was not trained on this frozen design')
        policy_path = DATA / record['policy']['dataset_relative_path']
        if digest(policy_path) != record['policy']['sha256']:
            raise ValueError('Frozen policy changed')
        POLICY = PPO.load(policy_path, device='cpu')


def rollout(condition):
    observation, _ = ENV.reset(
        seed=condition['seed'], options={'condition': condition})
    actions = []
    started = time.monotonic()
    for step in range(400):
        if VARIANT == 'B0':
            action = np.zeros(2)
        elif VARIANT == 'B1':
            action = search_action(step)
        else:
            action = POLICY.predict(observation, deterministic=True)[0]
        observation, _, done, _, info = ENV.step(action)
        actions.append(np.asarray(action).tolist())
        if done:
            row = {
                'condition_id': condition['condition_id'],
                'variant': VARIANT,
                'training_seed': POLICY_SEED,
                'wall_s': time.monotonic() - started,
                **info,
            }
            destination = OUTPUT_ROOT / (
                f'{VARIANT}_seed_{POLICY_SEED}' if VARIANT == 'B2' else VARIANT)
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / f"{condition['condition_id']}.json"
            path.write_text(json.dumps({**row, 'actions': actions}, indent=2) + '\n')
            row['trace'] = {
                'dataset_relative_path': str(path.relative_to(DATA)),
                'sha256': digest(path),
            }
            return row
    raise RuntimeError('Full-task held-out rollout did not terminate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=('B0', 'B1', 'B2'))
    parser.add_argument('--policy-seed', type=int)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if (args.variant == 'B2') != (args.policy_seed is not None):
        parser.error('B2 requires --policy-seed; B0/B1 do not use it')

    design_path = ROOT / 'experiments/residual_fulltask_mvp/contact_drift_design.json'
    design = json.loads(design_path.read_text())
    if args.policy_seed is not None and args.policy_seed not in design['training']['seeds']:
        parser.error('--policy-seed must be frozen in contact_drift_design.json')
    cases = heldout_conditions(design)
    output_root = DATA / 'self_evolution/residual_fulltask_mvp/contact_drift/heldout'
    rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize,
        initargs=(args.variant, args.policy_seed, design_path, output_root),
        mp_context=multiprocessing.get_context('spawn'),
    ) as pool:
        for row in pool.map(rollout, cases):
            rows.append(row)
            print('HELDOUT', row['condition_id'], args.variant, args.policy_seed,
                  row['is_success'], row['collision'], flush=True)

    run_name = f'B2_seed_{args.policy_seed}' if args.variant == 'B2' else args.variant
    payload = {
        'schema_version': 1,
        'benchmark': 'CONTACT_MEDIATED_DRIFT_V1',
        'variant': args.variant,
        'training_seed': args.policy_seed,
        'design_sha256': digest(design_path),
        'heldout_generator_seed': design['heldout']['generator_seed'],
        'conditions': cases,
        'cases': rows,
        'source_hashes': {
            str(path.relative_to(ROOT)): digest(path)
            for path in [
                Path(__file__), ROOT / 'robot_agent/full_task_residual_env.py',
                ROOT / 'robot_agent/full_task_observation.py',
                ROOT / 'robot_agent/residual_controller.py',
                ROOT / 'scripts/pilot_residual_v2.py',
            ]
        },
    }
    path = ROOT / 'experiments/residual_fulltask_mvp' / f'contact_drift_evaluation_{run_name}.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Frozen held-out cases and bootstrap-ready paired results; no policy tuning."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent.residual_env import LocalResidualEnv, DATA, ROOT
from pilot_residual_v2 import search_action
from learn import digest


def cases(design):
    result = []
    amplitudes = design['heldout']['socket_amplitude_mm']
    for i, seed in enumerate(design['heldout']['seeds']):
        rng = np.random.default_rng(seed)
        angle = (i % 8) * np.pi / 4
        amplitude = amplitudes[(i // 8) % 2]
        result.append(dict(id=f'heldout_{seed}', seed=seed, group='heldout', perturbation=dict(
            socket_xy_mm=(amplitude * np.array([np.cos(angle), np.sin(angle)])).tolist(),
            hand_xy_mm=rng.uniform(-.15, .15, 2).tolist(), yaw_deg=float(rng.uniform(-.75, .75)))))
        if design.get('actuator_stage'):
            result[-1]['perturbation'].update(controller_delay_frames=2,
                gripper_delta_mm=float(rng.choice([-.075, .075])))
    for seed in design['nominal_regression']['seeds']:
        rng = np.random.default_rng(seed)
        result.append(dict(id=f'nominal_{seed}', seed=seed, group='nominal', perturbation=dict(
            socket_xy_mm=[0., 0.], hand_xy_mm=rng.uniform(-.02, .02, 2).tolist(), yaw_deg=float(rng.uniform(-.1, .1)))))
    return result


def initialize(variant, dropout):
    global ENV, POLICY, VARIANT
    VARIANT = variant
    learned = variant not in ('B0', 'B1')
    history = 1 if variant.startswith('B2') else 15
    visual = variant.startswith('B3')
    ENV = LocalResidualEnv(history_frames=history, visual_features=visual, visual_dropout=dropout, training=False)
    POLICY = None
    if learned:
        from stable_baselines3 import PPO
        record = json.loads((ROOT / f'experiments/event_residual_mvp_v2/training_{variant}.json').read_text())
        path = DATA / record['policy']['dataset_relative_path']
        if digest(path) != record['policy']['sha256']:
            raise ValueError('Final policy changed after training')
        POLICY = PPO.load(path, device='cpu')


def rollout(case):
    obs, _ = ENV.reset(seed=case['seed'], options={'perturbation': case['perturbation']})
    actions, trace = [], []
    start = time.monotonic()
    for step in range(100):
        action = (np.zeros(2) if VARIANT == 'B0' else search_action(step) if VARIANT == 'B1'
                  else POLICY.predict(obs, deterministic=True)[0])
        obs, _, done, _, info = ENV.step(action)
        actions.append(np.asarray(action).tolist())
        trace.append(dict(frame=ENV.index, phase=ENV.player.frames[ENV.index].phase,
            nominal_action=ENV.player.frames[ENV.index].nominal_action, final_action=ENV.last_action.tolist(),
            residual_xy_m=ENV.smoother.position.tolist(), residual_velocity_m_s=ENV.smoother.velocity.tolist()))
        if done:
            result = dict(case_id=case['id'], seed=case['seed'], group=case['group'], variant=VARIANT,
                          visual_dropout=ENV.visual_dropout, wall_s=time.monotonic()-start, **info)
            destination = DATA / 'self_evolution/event_residual_mvp_v2/evaluation' / f'{VARIANT}_dropout{ENV.visual_dropout:g}'
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / (case['id'] + '.json')
            if path.exists():
                raise FileExistsError('Do not overwrite a completed heldout run')
            path.write_text(json.dumps(dict(**result, actions=actions, trace=trace), indent=2) + '\n')
            result['trace'] = dict(dataset_relative_path=str(path.relative_to(DATA)), sha256=digest(path))
            return result
    raise RuntimeError('Frozen local task did not finish')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=['B0', 'B1', 'B2_structured', 'B3_structured', 'B3_sparse', 'B3_actuator'])
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--phase2', action='store_true')
    args = parser.parse_args()
    design_path = ROOT / 'experiments/event_residual_mvp_v2' / ('phase2_design.json' if args.phase2 else 'design.json')
    design = json.loads(design_path.read_text())
    jobs = cases(design)
    if args.dropout == 1.:
        jobs = [case for case in jobs if case['group'] == 'heldout']
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
            initargs=(args.variant, args.dropout), mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(rollout, jobs):
            rows.append(row)
            print('HELDOUT', row['case_id'], row['variant'], row['is_success'], row['collision'], flush=True)
    result = dict(variant=args.variant, visual_dropout=args.dropout, design_sha256=digest(design_path),
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__), ROOT/'robot_agent/residual_env.py', ROOT/'robot_agent/residual_controller.py']}, cases=rows)
    suffix = '_phase2' if args.phase2 else ''
    path = ROOT / f'experiments/event_residual_mvp_v2/evaluation_{args.variant}_dropout{args.dropout:g}{suffix}.json'
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

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


def initialize(variant, dropout, run_tag, stage_a, policy_seed, evaluation_root):
    global ENV, POLICY, VARIANT, RUN_NAME, POLICY_SEED, EVALUATION_ROOT
    VARIANT = variant
    POLICY_SEED = policy_seed
    EVALUATION_ROOT = Path(evaluation_root)
    learned = variant not in ('B0', 'B1')
    history = 15 if stage_a else (1 if variant.startswith('B2') else 15)
    visual = False if stage_a else variant.startswith('B3')
    ENV = LocalResidualEnv(history_frames=history, visual_features=visual, visual_dropout=dropout, training=False)
    RUN_NAME = (f'{variant}_seed_{policy_seed}' if stage_a and learned else variant
                if stage_a else f'{variant}_dropout{ENV.visual_dropout:g}{run_tag}')
    POLICY = None
    if learned:
        from stable_baselines3 import PPO
        record_path = (ROOT / f'experiments/residual_fulltask_mvp/training_seed_{policy_seed}.json'
                       if stage_a else ROOT / f'experiments/event_residual_mvp_v2/training_{variant}.json')
        record = json.loads(record_path.read_text())
        if stage_a:
            design_path = ROOT / 'experiments/residual_fulltask_mvp/design.json'
            design = json.loads(design_path.read_text())
            expected = {k: v for k, v in design['training'].items() if k != 'seeds'}
            expected['seed'] = policy_seed
            if record['design_sha256'] != digest(design_path) or record['config'] != expected:
                raise ValueError('Stage A policy does not match its frozen seed and design')
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
                          training_seed=POLICY_SEED, visual_dropout=ENV.visual_dropout,
                          wall_s=time.monotonic()-start, **info)
            destination = EVALUATION_ROOT / RUN_NAME
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
    parser.add_argument('--variant', required=True, choices=['B0', 'B1', 'B2', 'B2_structured', 'B3_structured', 'B3_sparse', 'B3_actuator'])
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--phase2', action='store_true')
    parser.add_argument('--nominal-only', action='store_true')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--stage-a', action='store_true')
    parser.add_argument('--policy-seed', type=int)
    args = parser.parse_args()
    if args.stage_a:
        if args.variant not in ('B0', 'B1', 'B2'):
            parser.error('--stage-a uses B0, B1 or B2')
        if (args.variant == 'B2') != (args.policy_seed is not None):
            parser.error('Stage A B2 requires --policy-seed; baselines do not')
        if args.dropout or args.phase2 or args.nominal_only or args.run_tag:
            parser.error('Stage A evaluates its frozen full condition set without V2 modifiers')
    elif args.variant == 'B2' or args.policy_seed is not None:
        parser.error('B2 and --policy-seed belong to --stage-a')
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT/'robot_agent/residual_env.py',
        ROOT/'robot_agent/residual_controller.py', ROOT/'robot_agent/proprio_history.py', ROOT/'robot_agent/observation_adapter.py',
        ROOT/'robot_agent/nominal_reference.py', ROOT/'scripts/pilot_residual_v2.py']}
    run_name = f'{args.variant}_seed_{args.policy_seed}' if args.variant == 'B2' else args.variant
    snapshots = (DATA / 'self_evolution/residual_fulltask_mvp/stage_a/evaluation_sources' / run_name
                 if args.stage_a else DATA / 'self_evolution/event_residual_mvp_v2/evaluation_sources' /
                 f'{args.variant}_{args.dropout:g}_{args.phase2}_{args.nominal_only}{args.run_tag}')
    for relative in source_hashes:
        copy = snapshots / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((ROOT / relative).read_bytes())
    design_path = (ROOT / 'experiments/residual_fulltask_mvp/design.json' if args.stage_a else
                   ROOT / 'experiments/event_residual_mvp_v2' / ('phase2_design.json' if args.phase2 else 'design.json'))
    design = json.loads(design_path.read_text())
    if args.stage_a and args.policy_seed is not None and args.policy_seed not in design['training']['seeds']:
        parser.error('--policy-seed must be frozen in residual_fulltask_mvp/design.json')
    jobs = cases(design)
    if args.nominal_only:
        jobs = [case for case in jobs if case['group'] == 'nominal']
    if args.dropout == 1.:
        jobs = [case for case in jobs if case['group'] == 'heldout']
    rows = []
    evaluation_root = (DATA / 'self_evolution/residual_fulltask_mvp/stage_a/evaluation' if args.stage_a else
                       DATA / 'self_evolution/event_residual_mvp_v2/evaluation')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
            initargs=(args.variant, args.dropout, args.run_tag, args.stage_a, args.policy_seed, evaluation_root),
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(rollout, jobs):
            rows.append(row)
            print('HELDOUT', row['case_id'], row['variant'], row['is_success'], row['collision'], flush=True)
    result = dict(variant=args.variant, training_seed=args.policy_seed, visual_dropout=args.dropout,
        design_sha256=digest(design_path),
        source_hashes=source_hashes, source_capture='before rollout workers; exact new-source snapshots retained in DATA_ROOT', cases=rows)
    if args.stage_a:
        path = ROOT / 'experiments/residual_fulltask_mvp' / f'evaluation_{run_name}.json'
    else:
        suffix = '_phase2' if args.phase2 else ''
        if args.nominal_only: suffix += '_nominal'
        suffix += args.run_tag
        path = ROOT / f'experiments/event_residual_mvp_v2/evaluation_{args.variant}_dropout{args.dropout:g}{suffix}.json'
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

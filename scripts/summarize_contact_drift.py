"""Aggregate contact-drift training and paired held-out evidence."""
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/residual_fulltask_mvp'
sys.path.insert(0, str(ROOT / 'scripts'))
from learn import DATA, digest


def interval(values):
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def paired_bootstrap(delta, rng, samples=10000):
    delta = np.asarray(delta, dtype=float)
    draws = np.empty(samples)
    for index in range(samples):
        draws[index] = rng.choice(delta, len(delta), replace=True).mean()
    return interval(draws)


def hierarchical_bootstrap(delta, rng, samples=10000):
    delta = np.asarray(delta, dtype=float)
    draws = np.empty(samples)
    for index in range(samples):
        seeds = rng.integers(0, delta.shape[0], delta.shape[0])
        conditions = rng.integers(0, delta.shape[1], delta.shape[1])
        draws[index] = delta[np.ix_(seeds, conditions)].mean()
    return interval(draws)


def load_evaluation(name):
    path = EXPERIMENT / f'contact_drift_evaluation_{name}.json'
    return path, json.loads(path.read_text())


def summarize_cases(cases, b0):
    success = np.array([row['is_success'] for row in cases], dtype=bool)
    baseline = np.array([row['is_success'] for row in b0], dtype=bool)
    return {
        'successes': int(success.sum()),
        'conditions': len(cases),
        'success_rate': float(success.mean()),
        'recovered_b0_failures': int(np.sum(~baseline & success)),
        'spoiled_b0_successes': int(np.sum(baseline & ~success)),
        'collision_failures': int(sum(row['collision'] for row in cases)),
        'mean_correction_integral_m_s': float(np.mean([
            row['correction_integral_m_s'] for row in cases])),
        'mean_correction_variation_m': float(np.mean([
            row['correction_variation_m'] for row in cases])),
        'max_residual_m': float(max(row['max_residual_m'] for row in cases)),
        'max_residual_speed_m_s': float(max(row['max_residual_speed_m_s'] for row in cases)),
        'max_residual_acceleration_m_s2': float(max(
            row['max_residual_acceleration_m_s2'] for row in cases)),
        'max_joint_correction_rad': float(max(row['max_joint_correction_rad'] for row in cases)),
    }


def main():
    design_path = EXPERIMENT / 'contact_drift_design.json'
    pilot_path = EXPERIMENT / 'contact_drift_pilot.json'
    design = json.loads(design_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    design_hash = digest(design_path)

    training_rows = []
    for seed in design['training']['seeds']:
        path = EXPERIMENT / f'contact_drift_training_seed_{seed}.json'
        row = json.loads(path.read_text())
        policy_path = DATA / row['policy']['dataset_relative_path']
        episodes_path = policy_path.parent / 'episodes.json'
        episodes = json.loads(episodes_path.read_text())
        terminal = [episode for episode in episodes if 'is_success' in episode]
        training_rows.append({
            **row,
            'record_sha256': digest(path),
            'training_successes': sum(episode['is_success'] for episode in terminal),
            'training_terminal_episodes': len(terminal),
            'training_success_rate': (sum(episode['is_success'] for episode in terminal) / len(terminal)
                                      if terminal else None),
            'episodes_sha256': digest(episodes_path),
        })
    training_valid = all(
        row['design_sha256'] == design_hash
        and row['actual_steps'] == design['training']['steps_per_seed']
        and row['privileged_student_inputs'] == 0
        for row in training_rows)
    training_payload = {
        'schema_version': 1,
        'benchmark': 'CONTACT_MEDIATED_DRIFT_V1',
        'stage': 'TRAINING_COMPLETE',
        'validity': 'VALID' if training_valid else 'INVALID',
        'design_sha256': design_hash,
        'fixed_budget_checkpoint_selection': 'final checkpoint only; no held-out tuning',
        'seeds': training_rows,
        'summary': {
            'seeds': len(training_rows),
            'actual_steps': sum(row['actual_steps'] for row in training_rows),
            'wall_s_max_parallel': max(row['wall_s'] for row in training_rows),
            'policies': len(training_rows),
            'privileged_student_inputs': 0,
        },
        'permissions': design['permissions'],
    }
    training_path = EXPERIMENT / 'contact_drift_training.json'
    training_path.write_text(json.dumps(training_payload, indent=2) + '\n')

    b0_path, b0_payload = load_evaluation('B0')
    b1_path, b1_payload = load_evaluation('B1')
    b2_payloads = []
    evaluation_paths = [b0_path, b1_path]
    for seed in design['training']['seeds']:
        path, payload = load_evaluation(f'B2_seed_{seed}')
        evaluation_paths.append(path)
        b2_payloads.append(payload)
    evaluations = [b0_payload, b1_payload, *b2_payloads]
    condition_ids = [row['condition_id'] for row in b0_payload['cases']]
    paired = all(
        payload['design_sha256'] == design_hash
        and [row['condition_id'] for row in payload['cases']] == condition_ids
        for payload in evaluations)
    b0 = b0_payload['cases']
    b1 = b1_payload['cases']
    b2 = [payload['cases'] for payload in b2_payloads]
    b0_success = np.asarray([row['is_success'] for row in b0], dtype=float)
    b1_success = np.asarray([row['is_success'] for row in b1], dtype=float)
    b2_success = np.asarray([
        [row['is_success'] for row in cases] for cases in b2], dtype=float)
    rng = np.random.default_rng(design['heldout']['bootstrap_seed'])
    b1_delta = b1_success - b0_success
    b2_delta = b2_success - b0_success[None, :]
    b0_summary = summarize_cases(b0, b0)
    b1_summary = summarize_cases(b1, b0)
    b2_summaries = [summarize_cases(cases, b0) for cases in b2]
    for seed, summary, delta in zip(design['training']['seeds'], b2_summaries, b2_delta):
        summary['training_seed'] = seed
        summary['improvement_over_b0_pp'] = float(delta.mean() * 100)
        summary['paired_ci95_pp'] = [value * 100 for value in paired_bootstrap(delta, rng)]
    mean_b2_rate = float(b2_success.mean())
    mean_b2_delta = float(b2_delta.mean())
    b2_ci = [value * 100 for value in hierarchical_bootstrap(b2_delta, rng)]
    seed_rates = b2_success.mean(axis=1)
    action_bounds_valid = all(
        summary['max_residual_m'] <= .002000001
        and summary['max_residual_speed_m_s'] <= .003000001
        and summary['max_residual_acceleration_m_s2'] <= .030000001
        and summary['max_joint_correction_rad'] <= .030000001
        for summary in [b1_summary, *b2_summaries])
    isolation_valid = all(
        row['privileged_student_inputs'] == 0
        for payload in evaluations for row in payload['cases'])
    integrity_valid = training_valid and paired and action_bounds_valid and isolation_valid
    positive_seeds = sum(summary['improvement_over_b0_pp'] > 0 for summary in b2_summaries)
    if not integrity_valid:
        finding = 'INVALID'
    elif positive_seeds == len(b2_summaries) and b2_ci[0] > 0:
        finding = 'STRONG_SUPPORT'
    elif mean_b2_delta > 0 and positive_seeds >= 2:
        finding = 'LIMITED_SUPPORT'
    else:
        finding = 'NOT_SUPPORTED'

    heldout_payload = {
        'schema_version': 1,
        'benchmark': 'CONTACT_MEDIATED_DRIFT_V1',
        'stage': 'HELDOUT_COMPLETE',
        'validity': 'VALID' if integrity_valid else 'INVALID',
        'finding': finding,
        'design_sha256': design_hash,
        'pilot_sha256': digest(pilot_path),
        'conditions': b0_payload['conditions'],
        'summaries': {
            'B0_frozen_nominal': b0_summary,
            'B1_deterministic_search': {
                **b1_summary,
                'improvement_over_b0_pp': float(b1_delta.mean() * 100),
                'paired_ci95_pp': [value * 100 for value in paired_bootstrap(b1_delta, rng)],
            },
            'B2_history_ppo_by_seed': b2_summaries,
            'B2_history_ppo_mean': {
                'success_rate': mean_b2_rate,
                'improvement_over_b0_pp': mean_b2_delta * 100,
                'hierarchical_paired_ci95_pp': b2_ci,
                'training_seed_success_rate_variance': float(np.var(seed_rates, ddof=1)),
                'positive_training_seeds': positive_seeds,
                'recovered_b0_failures_mean': float(np.mean([
                    summary['recovered_b0_failures'] for summary in b2_summaries])),
                'spoiled_b0_successes_mean': float(np.mean([
                    summary['spoiled_b0_successes'] for summary in b2_summaries])),
                'collision_failures_total': sum(
                    summary['collision_failures'] for summary in b2_summaries),
            },
        },
        'integrity': {
            'paired_identical_conditions': paired,
            'action_bounds_valid': action_bounds_valid,
            'privileged_student_inputs': 0 if isolation_valid else 'VIOLATION',
            'heldout_tuning': 0,
            'hardware_executions': 0,
            'real_data_reads': 0,
        },
        'raw_evaluations': {
            path.stem.removeprefix('contact_drift_evaluation_'): {
                'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
            for path in evaluation_paths
        },
    }
    heldout_path = EXPERIMENT / 'contact_drift_heldout.json'
    heldout_path.write_text(json.dumps(heldout_payload, indent=2) + '\n')

    mean = heldout_payload['summaries']['B2_history_ppo_mean']
    b1s = heldout_payload['summaries']['B1_deterministic_search']
    lines = [
        '# CONTACT_MEDIATED_DRIFT_V1 Residual Report',
        '',
        f"**Finding: {finding}.**",
        '',
        'The core experiment used the frozen full Episode, unchanged physics, a contact-mediated lift disturbance, and the deployable P0 history only. Requested disturbance direction and privileged drift were not Student inputs.',
        '',
        '## Frozen challenge',
        '',
        f"The distribution was frozen before training: lift-middle, {design['distribution']['force_n']:g} N for {design['distribution']['duration_ms']:g} ms, with angle uniform over {design['distribution']['angle_rad']['intervals']}. The formal pilot had {pilot['summary']['successes']}/{pilot['summary']['valid_conditions']} Frozen-Nominal successes and no rejected condition inside the final distribution." ,
        '',
        '## Paired held-out results',
        '',
        '| Method | Success | Improvement vs B0 | Paired 95% CI | Recovered | Spoiled | Collisions |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| B0 Frozen Nominal | {b0_summary['successes']}/{b0_summary['conditions']} | — | — | — | — | {b0_summary['collision_failures']} |",
        f"| B1 Deterministic Search | {b1_summary['successes']}/{b1_summary['conditions']} | {b1s['improvement_over_b0_pp']:.2f} pp | [{b1s['paired_ci95_pp'][0]:.2f}, {b1s['paired_ci95_pp'][1]:.2f}] pp | {b1_summary['recovered_b0_failures']} | {b1_summary['spoiled_b0_successes']} | {b1_summary['collision_failures']} |",
    ]
    for summary in b2_summaries:
        lines.append(
            f"| B2 PPO seed {summary['training_seed']} | {summary['successes']}/{summary['conditions']} | {summary['improvement_over_b0_pp']:.2f} pp | [{summary['paired_ci95_pp'][0]:.2f}, {summary['paired_ci95_pp'][1]:.2f}] pp | {summary['recovered_b0_failures']} | {summary['spoiled_b0_successes']} | {summary['collision_failures']} |")
    lines += [
        f"| B2 PPO mean | {mean['success_rate'] * 100:.2f}% | {mean['improvement_over_b0_pp']:.2f} pp | [{mean['hierarchical_paired_ci95_pp'][0]:.2f}, {mean['hierarchical_paired_ci95_pp'][1]:.2f}] pp | {mean['recovered_b0_failures_mean']:.2f} | {mean['spoiled_b0_successes_mean']:.2f} | {mean['collision_failures_total']} total |",
        '',
        '## Integrity and interpretation',
        '',
        f"All {len(training_rows)} policies are fixed-budget final checkpoints. Paired-condition integrity: {paired}. Action bounds: {action_bounds_valid}. Privileged Student inputs: 0. Hardware executions: 0. Held-out tuning: 0.",
        '',
        'The support label follows the amendment directly: strong support requires a consistent positive effect with a positive paired interval; limited support permits a positive but seed-sensitive effect; no reliable improvement is not supported. No arbitrary percentage-point threshold was imposed.',
        '',
    ]
    (EXPERIMENT / 'CONTACT_DRIFT_RESIDUAL_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'finding': finding, 'training': training_payload['summary'],
                      'heldout': heldout_payload['summaries']}, indent=2))


if __name__ == '__main__':
    main()

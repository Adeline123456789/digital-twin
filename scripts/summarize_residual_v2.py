"""Paired statistics, stopped-only capsules and action audits from frozen evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learn import DATA, ROOT, digest
from robot_agent.diagnosis import EventBuilder
from robot_agent.fast_loop import FastLoop
from robot_agent.memory_router import IsolatedExperimentMemory
from robot_agent.supervisor import Supervisor


def paired_bootstrap(base, candidate, seed=7721, resamples=10000):
    base, candidate = np.asarray(base, float), np.asarray(candidate, float)
    if base.shape != candidate.shape or base.ndim != 1 or not len(base):
        raise ValueError('Expected matched nonempty condition vectors')
    delta = candidate - base
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(axis=1)
    return dict(improvement_pp=float(delta.mean() * 100), ci95_pp=(np.quantile(means, [.025, .975]) * 100).tolist(),
                recovered=int(np.sum((base == 0) & (candidate == 1))), spoiled=int(np.sum((base == 1) & (candidate == 0))),
                recovery_rate=float(np.sum((base == 0) & (candidate == 1)) / max(1, np.sum(base == 0))))


def hierarchical_paired_bootstrap(base, candidates, seed=15101, resamples=10000):
    base, candidates = np.asarray(base, float), np.asarray(candidates, float)
    if base.ndim != 1 or candidates.ndim != 2 or candidates.shape[1] != len(base) or not candidates.size:
        raise ValueError('Expected conditions and training-seed by condition matrix')
    delta = candidates - base
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, len(delta), (resamples, len(delta), 1))
    condition_draws = rng.integers(0, len(base), (resamples, 1, len(base)))
    means = delta[seed_draws, condition_draws].mean(axis=(1, 2))
    return dict(improvement_pp=float(delta.mean() * 100),
                ci95_pp=(np.quantile(means, [.025, .975]) * 100).tolist(),
                per_training_seed_improvement_pp=(delta.mean(axis=1) * 100).tolist())


def audit_actions(row, frozen_frames):
    path = DATA / row['trace']['dataset_relative_path']
    if digest(path) != row['trace']['sha256']: raise ValueError('Evaluation trace changed')
    raw = json.loads(path.read_text())
    trace = raw['trace']
    command = np.array([r['final_action'] for r in trace])
    reference = np.array([frozen_frames[r['frame']]['nominal_action'] for r in trace])
    xy = np.array([r['residual_xy_m'] for r in trace])
    velocity = np.array([r['residual_velocity_m_s'] for r in trace])
    valid = (np.array_equal([r['nominal_action'] for r in trace], reference)
             and np.array_equal(command[:, :7], reference[:, :7]) and np.array_equal(command[:, 13], reference[:, 13])
             and np.max(abs(xy)) <= .002 + 1e-12 and np.max(abs(velocity)) <= .003 + 1e-12
             and np.max(abs(command[:, 7:13] - reference[:, 7:13])) <= .03 + 1e-12)
    return dict(variant=row['variant'], case_id=row['case_id'], valid=bool(valid),
        max_residual_axis_m=float(np.max(abs(xy))), max_residual_axis_speed_m_s=float(np.max(abs(velocity))),
        max_joint_correction_rad=float(np.max(abs(command[:, 7:13]-reference[:, 7:13]))))


def summarize_stage_a():
    folder = ROOT / 'experiments/residual_fulltask_mvp'
    design_path = folder / 'design.json'
    design = json.loads(design_path.read_text())
    design_hash = digest(design_path)
    evaluations = {
        'B0': json.loads((folder / 'evaluation_B0.json').read_text()),
        'B1': json.loads((folder / 'evaluation_B1.json').read_text()),
        **{seed: json.loads((folder / f'evaluation_B2_seed_{seed}.json').read_text())
           for seed in design['training']['seeds']},
    }
    if any(value['design_sha256'] != design_hash for value in evaluations.values()):
        raise ValueError('Stage A evaluation design differs from frozen design')

    def rows(value, group):
        return sorted((row for row in value['cases'] if row['group'] == group), key=lambda row: row['seed'])

    base = {group: rows(evaluations['B0'], group) for group in ('heldout', 'nominal')}
    search = {group: rows(evaluations['B1'], group) for group in ('heldout', 'nominal')}
    policies = {group: [rows(evaluations[seed], group) for seed in design['training']['seeds']]
                for group in ('heldout', 'nominal')}
    for group in ('heldout', 'nominal'):
        expected = [(row['case_id'], row['perturbation']) for row in base[group]]
        if [(row['case_id'], row['perturbation']) for row in search[group]] != expected:
            raise ValueError(f'Unpaired B1 {group} conditions')
        if any([(row['case_id'], row['perturbation']) for row in policy] != expected
               for policy in policies[group]):
            raise ValueError(f'Unpaired B2 {group} conditions')

    heldout_base = np.array([row['is_success'] for row in base['heldout']], float)
    heldout_search = np.array([row['is_success'] for row in search['heldout']], float)
    heldout_policies = np.array([[row['is_success'] for row in policy] for policy in policies['heldout']], float)
    nominal_base = np.array([row['is_success'] for row in base['nominal']], float)
    nominal_policies = np.array([[row['is_success'] for row in policy] for policy in policies['nominal']], float)
    bootstrap_seed = design['heldout']['bootstrap_seed']
    b2_vs_b0 = hierarchical_paired_bootstrap(heldout_base, heldout_policies, bootstrap_seed)
    b2_vs_b1 = hierarchical_paired_bootstrap(heldout_search, heldout_policies, bootstrap_seed)
    nominal = hierarchical_paired_bootstrap(nominal_base, nominal_policies, bootstrap_seed)
    b1_vs_b0 = paired_bootstrap(heldout_base, heldout_search, bootstrap_seed)

    nominal_manifest = json.loads((ROOT / 'experiments/event_residual_mvp/nominal_manifest.json').read_text())
    frozen_path = DATA / nominal_manifest['dataset_relative_path']
    if digest(frozen_path) != nominal_manifest['sha256']:
        raise ValueError('Nominal bytes changed')
    frozen_frames = json.loads(frozen_path.read_text())['frames']
    audits = [audit_actions(row, frozen_frames) for value in evaluations.values() for row in value['cases']]
    action_audit = dict(samples=len(audits), all_bounds_and_left_arm_invariance=all(row['valid'] for row in audits),
        max_residual_axis_m=max(row['max_residual_axis_m'] for row in audits),
        max_residual_axis_speed_m_s=max(row['max_residual_axis_speed_m_s'] for row in audits),
        max_joint_correction_rad=max(row['max_joint_correction_rad'] for row in audits))
    privileged_student_inputs = max(row['privileged_student_inputs']
                                    for value in evaluations.values() for row in value['cases'])

    positive_seeds = sum(value > 0 for value in b2_vs_b0['per_training_seed_improvement_pp'])
    regression_pp = -nominal['improvement_pp']
    gate = design['gate']
    criteria = dict(
        effect_size=b2_vs_b0['improvement_pp'] >= gate['mean_improvement_over_B0_pp_min'],
        ci_supported=b2_vs_b0['ci95_pp'][0] > gate['paired_hierarchical_ci95_lower_pp_min_exclusive'],
        seed_stability=positive_seeds >= gate['positive_training_seeds_min'],
        nominal_regression=regression_pp <= gate['nominal_regression_pp_max'],
        action_bounds=action_audit['all_bounds_and_left_arm_invariance'],
        privileged_student_inputs=privileged_student_inputs == 0)
    if all(criteria.values()):
        decision = 'GO'
    elif (criteria['effect_size'] and criteria['nominal_regression'] and criteria['action_bounds']
          and criteria['privileged_student_inputs']):
        decision = 'INCONCLUSIVE'
    else:
        decision = 'NO-GO'

    training = [json.loads((folder / f'training_seed_{seed}.json').read_text()) for seed in design['training']['seeds']]
    result = dict(stage='A', decision=decision, proceed_to_stage_b=decision == 'GO', criteria=criteria,
        statistics=dict(
            B0=dict(heldout_successes=int(heldout_base.sum()), heldout_n=len(heldout_base),
                    nominal_successes=int(nominal_base.sum()), nominal_n=len(nominal_base)),
            B1=dict(heldout_successes=int(heldout_search.sum()), heldout_n=len(heldout_search),
                    vs_B0=b1_vs_b0),
            B2=dict(training_seeds=design['training']['seeds'],
                    heldout_successes_per_seed=heldout_policies.sum(axis=1).astype(int).tolist(),
                    nominal_successes_per_seed=nominal_policies.sum(axis=1).astype(int).tolist(),
                    positive_training_seeds=positive_seeds, vs_B0=b2_vs_b0, vs_B1=b2_vs_b1,
                    nominal_regression_pp=regression_pp)),
        action_audit=action_audit, privileged_student_inputs=privileged_student_inputs,
        budget=dict(training_policy_steps=sum(record['actual_steps'] for record in training),
                    training_rollouts=sum(record['episodes'] for record in training),
                    evaluation_rollouts=sum(len(value['cases']) for value in evaluations.values()),
                    hardware=0, real_data_reads=0, real_training=0, runtime_llm=0, twin_mutations=0),
        inference_scope='Five fixed-budget training seeds; 64 paired held-out and 20 paired nominal conditions; hierarchical bootstrap over training seeds and conditions',
        candidate_mapping='Stage A B2 is the historical V2 B3_structured 15-frame architecture without RGB channels')
    (folder / 'reproduction.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    folder = ROOT / 'experiments/event_residual_mvp_v2'
    design = json.loads((folder / 'design.json').read_text())
    nominal = json.loads((ROOT / 'experiments/event_residual_mvp/nominal_manifest.json').read_text())
    frozen_path = DATA / nominal['dataset_relative_path']
    if digest(frozen_path) != nominal['sha256']: raise ValueError('Nominal bytes changed')
    frozen_frames = json.loads(frozen_path.read_text())['frames']
    variants = design['heldout']['policies']
    evaluations = {v: json.loads((folder / f"evaluation_{v}_dropout0{'_recheck' if v == 'B0' else ''}.json").read_text()) for v in variants}
    dropout = json.loads((folder / 'evaluation_B3_structured_dropout1_recheck.json').read_text())
    training = {v: json.loads((folder / f'training_{v}.json').read_text()) for v in variants if v not in ('B0', 'B1')}
    stats, audits, capsules = {}, [], []
    builder = EventBuilder()
    for variant, evaluation in [*evaluations.items(), ('B3_dropout', dropout)]:
        if evaluation['design_sha256'] != digest(folder / 'design.json'):
            raise ValueError('Evaluation design differs from frozen design')
        groups = {}
        for group in ('heldout', 'nominal'):
            rows = sorted([r for r in evaluation['cases'] if r['group'] == group], key=lambda r: r['seed'])
            if not rows: continue
            base = sorted([r for r in evaluations['B0']['cases'] if r['group'] == group], key=lambda r: r['seed'])
            if [r['seed'] for r in rows] != [r['seed'] for r in base]:
                raise ValueError('Unpaired evaluation seeds')
            if [r['perturbation'] for r in rows] != [r['perturbation'] for r in base]:
                raise ValueError('Paired initial perturbations differ')
            groups[group] = dict(n=len(rows), successes=sum(r['is_success'] for r in rows),
                success_rate=float(np.mean([r['is_success'] for r in rows])), collisions=sum(r['collision'] for r in rows),
                mean_correction_integral_m_s=float(np.mean([r['correction_integral_m_s'] for r in rows])),
                mean_correction_variation_m=float(np.mean([r['correction_variation_m'] for r in rows])),
                mean_duration_s=float(np.mean([r['duration_s'] for r in rows])),
                **paired_bootstrap([r['is_success'] for r in base], [r['is_success'] for r in rows]))
        stats[variant] = groups
        for row in evaluation['cases']:
            path = DATA / row['trace']['dataset_relative_path']
            audits.append(audit_actions(row, frozen_frames))
            if not row['is_success']:
                policy_hash = (training[row['variant']]['policy']['sha256'] if row['variant'] in training
                               else nominal['policy_hash'])
                report = builder.stopped_local(path, nominal, policy_hash)
                supervisor = Supervisor(fast=FastLoop(router=IsolatedExperimentMemory()))
                answer = supervisor.diagnose(report)
                capsules.append(dict(case_id=row['case_id'], variant=variant, **answer,
                    slow_calls=supervisor.calls, terminal_only=True, physical_onset='UNKNOWN'))
    audit_result = dict(samples=len(audits), all_bounds_and_left_arm_invariance=all(r['valid'] for r in audits),
        max_residual_axis_m=max(r['max_residual_axis_m'] for r in audits),
        max_residual_axis_speed_m_s=max(r['max_residual_axis_speed_m_s'] for r in audits),
        scope='15 Hz recorded actions; 30 Hz filter speed/acceleration bound independently covered by unit tests')
    b3 = stats['B3_structured']; improvement = b3['heldout']; regression = -b3['nominal']['improvement_pp']
    supported = improvement['ci95_pp'][0] > 0
    go = supported and regression <= 5 and audit_result['all_bounds_and_left_arm_invariance']
    decision = 'GO' if go else ('INCONCLUSIVE' if improvement['improvement_pp'] > 0 and regression <= 5 else 'NO-GO')
    output = dict(decision=decision, statistics=stats, action_audit=audit_result,
        inference_scope='32 paired heldout conditions, single trained seed per architecture; bootstrap conditions, not training seeds',
        no_runtime_llm=True, no_hardware=True, no_real_training=True, no_automatic_twin_adaptation=True,
        runtime_fixture_perturbations='predeclared physical geometry reset perturbations; not adaptive Twin changes')
    (folder / 'heldout_results.json').write_text(json.dumps(output, indent=2) + '\n')
    equal = []
    normal = {r['case_id']:r for r in evaluations['B3_structured']['cases'] if r['group']=='heldout'}
    for row in dropout['cases']:
        a = json.loads((DATA / row['trace']['dataset_relative_path']).read_text())['actions']
        b = json.loads((DATA / normal[row['case_id']]['trace']['dataset_relative_path']).read_text())['actions']
        equal.append(a == b)
    rgb = dict(dropout_0=stats['B3_structured']['heldout'], dropout_1=stats['B3_dropout']['heldout'],
        identical_action_sequences=sum(equal), paired_conditions=len(equal),
        qualified_vision_available=False, limitation='Optional visual branch uses unqualified RGB candidates. If absent, dropout equivalence demonstrates proprioceptive execution, not useful visual fusion.',
        raw_visual_available_samples=sum(r['visual_available_samples'] for r in evaluations['B3_structured']['cases']),
        raw_visual_samples=sum(r['visual_samples'] for r in evaluations['B3_structured']['cases']))
    (folder / 'rgb_dropout_results.json').write_text(json.dumps(rgb, indent=2) + '\n')
    (folder / 'residual_training.json').write_text(json.dumps(dict(runs=training, total_steps=sum(r['actual_steps'] for r in training.values()),
        train_range=design['training_range'], nominal_updated=False, reward_ablation='structured vs sparse, same B3 architecture/seed/step budget'), indent=2) + '\n')
    raw = DATA / 'self_evolution/event_residual_mvp_v2/offline_capsules.json'
    raw.write_text(json.dumps(capsules, indent=2) + '\n')
    (folder / 'offline_diagnosis.json').write_text(json.dumps(dict(failed_rollouts=len(capsules), slow_calls=sum(c['slow_calls'] for c in capsules),
        legacy_skill_invocations=0, root_cause='UNKNOWN', evidence=dict(dataset_relative_path=str(raw.relative_to(DATA)), sha256=digest(raw)),
        stage='Stopped-only EventBuilder/FailureReport/Supervisor capsules, isolated memory; no structural proposal or Twin edit'), indent=2) + '\n')
    phase2 = {v: json.loads((folder / f"evaluation_{v}_dropout0_phase2{'_recheck' if v != 'B3_actuator' else ''}.json").read_text()) for v in ('B0', 'B3_structured', 'B3_actuator')}
    phase2_stats = {}
    baseline_rows = sorted(phase2['B0']['cases'], key=lambda r: r['seed'])
    for variant, value in phase2.items():
        rows = sorted(value['cases'], key=lambda r: r['seed'])
        if value['design_sha256'] != digest(folder / 'phase2_design.json') or [r['perturbation'] for r in rows] != [r['perturbation'] for r in baseline_rows]:
            raise ValueError('Unpaired phase2 design or conditions')
        phase2_stats[variant] = dict(n=len(rows), successes=sum(r['is_success'] for r in rows), collisions=sum(r['collision'] for r in rows),
            **paired_bootstrap([r['is_success'] for r in baseline_rows], [r['is_success'] for r in rows]))
    nominal2 = json.loads((folder / 'evaluation_B3_actuator_dropout0_nominal.json').read_text())['cases']
    nominal_base = sorted([r for r in evaluations['B0']['cases'] if r['group']=='nominal'], key=lambda r:r['seed'])
    nominal2 = sorted(nominal2, key=lambda r:r['seed'])
    if [r['perturbation'] for r in nominal2] != [r['perturbation'] for r in nominal_base]: raise ValueError('Unpaired actuator nominal regression')
    regression2 = paired_bootstrap([r['is_success'] for r in nominal_base], [r['is_success'] for r in nominal2])
    phase2_result = dict(statistics=phase2_stats, nominal_regression=dict(n=len(nominal2), successes=sum(r['is_success'] for r in nominal2), **regression2),
        scope='Separate sequential actuator curriculum and new heldout seeds; original geometry-only conclusion retained',
        decision=('GO' if phase2_stats['B3_actuator']['ci95_pp'][0] > 0 else 'INCONCLUSIVE')
                 if regression2['improvement_pp'] >= -5 else 'NO-GO', real_transfer_allowed=False)
    phase2_audits = [audit_actions(row, frozen_frames) for evaluation in phase2.values() for row in evaluation['cases']]
    phase2_audits += [audit_actions(row, frozen_frames) for row in nominal2]
    phase2_result['action_audit'] = dict(samples=len(phase2_audits), all_bounds_and_left_arm_invariance=all(r['valid'] for r in phase2_audits),
        max_residual_axis_m=max(r['max_residual_axis_m'] for r in phase2_audits),
        max_residual_axis_speed_m_s=max(r['max_residual_axis_speed_m_s'] for r in phase2_audits))
    if not phase2_result['action_audit']['all_bounds_and_left_arm_invariance']:
        phase2_result['decision'] = 'NO-GO'
    (folder / 'phase2_results.json').write_text(json.dumps(phase2_result, indent=2) + '\n')
    training['B3_actuator'] = json.loads((folder / 'training_B3_actuator.json').read_text())
    (folder / 'residual_training.json').write_text(json.dumps(dict(runs=training, total_steps=sum(r['actual_steps'] for r in training.values()),
        train_range=design['training_range'], nominal_updated=False, reward_ablation='structured vs sparse with matched B3 architecture/seed/step budget',
        phase2='Additional residual-only curriculum: 0–1 frame actuator delay, +/-0.05 mm initial aperture; separate 2-frame/+/-0.075 mm heldout conditions'), indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-a', action='store_true')
    args = parser.parse_args()
    summarize_stage_a() if args.stage_a else main()

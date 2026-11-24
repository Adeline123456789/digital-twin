"""Audit controlled post-grasp XY perturbations and deployable observability."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug')
EXPERIMENT = ROOT / 'experiments/residual_fulltask_mvp'
SOURCE_ID = '20261123T090549Z_a35ab76b'
DIRECTIONS = {
    'positive_x': np.array([1., 0.]),
    'negative_x': np.array([-1., 0.]),
    'positive_y': np.array([0., 1.]),
    'negative_y': np.array([0., -1.]),
}


def area_under_roc(positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + .5 * (comparisons == 0)))


def ridge_xy(train_x, train_y, test_x, penalty=1e-4):
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale < 1e-8] = 1.
    train = np.c_[(train_x - mean) / scale, np.ones(len(train_x))]
    test = np.c_[(test_x - mean) / scale, np.ones(len(test_x))]
    weights = train.T @ np.linalg.solve(train @ train.T + penalty * np.eye(len(train)), train_y)
    return test @ weights


def cardinal_direction(xy):
    names = tuple(DIRECTIONS)
    matrix = np.stack([DIRECTIONS[name] for name in names])
    return names[int(np.argmax(np.asarray(xy) @ matrix.T))]


def array_hash(value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    return hashlib.sha256(value.tobytes()).hexdigest()


def evaluate_xy(train_features, train_nominal, train_targets,
                heldout_features, heldout_nominal, heldout_targets,
                current_directions, treatment_directions):
    prediction = ridge_xy(
        np.r_[train_features, train_nominal],
        np.r_[train_targets, np.zeros_like(train_targets)],
        np.r_[heldout_features, heldout_nominal],
    )
    perturbed, nominal = np.split(prediction, 2)
    direction_names = tuple(DIRECTIONS)
    direction_matrix = np.stack([DIRECTIONS[name] for name in direction_names])
    predicted_names = [direction_names[index] for index in np.argmax(perturbed @ direction_matrix.T, axis=1)]
    current_recall = []
    for name in direction_names:
        mask = np.asarray(current_directions) == name
        current_recall.append(float(np.mean(np.asarray(predicted_names)[mask] == name)) if mask.any() else 0.)
    current_macro_recall = float(np.mean(current_recall))
    treatment_accuracy = float(
        np.mean(np.asarray(predicted_names) == np.asarray(treatment_directions)))
    return {
        'detection_auroc': area_under_roc(np.linalg.norm(perturbed, axis=1),
                                          np.linalg.norm(nominal, axis=1)),
        'direction_accuracy': min(current_macro_recall, treatment_accuracy),
        'current_direction_accuracy': float(
            np.mean(np.asarray(predicted_names) == np.asarray(current_directions))),
        'current_direction_macro_recall': current_macro_recall,
        'current_direction_recall_by_class': dict(zip(direction_names, current_recall)),
        'treatment_direction_accuracy': treatment_accuracy,
        'xy_rmse_mm': float(np.sqrt(np.mean((perturbed - heldout_targets) ** 2))),
        'predicted_xy_mm': perturbed.tolist(),
        'nominal_predicted_xy_mm': nominal.tolist(),
        'predicted_directions': predicted_names,
    }


class SimulatorAudit:
    def __init__(self):
        import mujoco
        from learn import SIM, digest
        from robot_agent.nominal_reference import ReferencePlayer

        sys.path.insert(0, str(SIM / 'src'))
        from piper_sim.backends.mujoco import MujocoBackend
        from piper_sim.config import load
        from piper_sim.mixed_recording import configure_rigid_collection, checked_physics_step

        manifest = json.loads((ROOT / 'experiments/event_residual_mvp/nominal_manifest.json').read_text())
        reference_path = DATA / manifest['dataset_relative_path']
        self.player = ReferencePlayer(json.loads(reference_path.read_text()), manifest['reference_hash'])
        self.raw = DATA / 'self_evolution' / SOURCE_ID
        self.stats = DATA / 'complete_sim/succeed' / SOURCE_ID / 'proprio_stats'
        task = yaml.safe_load((self.raw / 'task.yaml').read_text())
        self.sim = MujocoBackend(dict(load(), task_scene=task), cameras=False)
        self.profile = configure_rigid_collection(self.sim)
        self.checked_step = checked_physics_step
        self.mujoco = mujoco
        self.grasp_end = max(i for i, frame in enumerate(self.player.frames) if frame.phase == 'grasp')
        privileged = json.loads(next(line for i, line in enumerate((self.stats / 'privileged.jsonl').open())
                                     if i == self.grasp_end))
        self.snapshot = privileged['physics_state']
        self.plug_qadr = self.sim.model.joint('task/yellow_plug/free').qposadr[0]
        self.plug_body = self.sim.model.body('task/yellow_plug').id
        self.right = self.sim.indices['right']
        self.source = {
            'episode': SOURCE_ID,
            'grasp_established_frame': self.grasp_end,
            'first_lift_frame': self.grasp_end + 1,
            'frozen_reference_sha256': digest(reference_path),
            'physics_state_mujoco_version': self.snapshot['mujoco_version'],
            'physics_state_spec': self.snapshot['spec'],
            'model_sha256': digest(self.stats / 'model.xml'),
        }

    def close(self):
        self.sim.close()

    def restore(self):
        self.mujoco.mj_setState(self.sim.model, self.sim.data,
                                np.asarray(self.snapshot['values']), self.snapshot['spec'])
        self.mujoco.mj_forward(self.sim.model, self.sim.data)

    def relation(self):
        grip = self.sim.data.body('right/gripper_center')
        plug = self.sim.data.body('task/yellow_plug')
        return grip.xmat.reshape(3, 3).T @ (plug.xpos - grip.xpos)

    def penetration(self):
        return max(0., -min((contact.dist for contact in self.sim.data.contact), default=0.))

    def shift(self, xy_mm):
        rotation = self.sim.data.body('right/gripper_center').xmat.reshape(3, 3)
        self.sim.data.qpos[self.plug_qadr:self.plug_qadr + 3] += rotation @ np.r_[np.asarray(xy_mm) / 1000, 0.]
        self.mujoco.mj_forward(self.sim.model, self.sim.data)

    def step(self, index, enforce_clock=True):
        action = np.asarray(self.player.frames[index].nominal_action)
        self.sim.apply(self.sim.layout.command(action, self.sim.sim_time, index))
        check = self.checked_step(self.sim, .0002)
        if enforce_clock and abs(self.sim.sim_time - self.player.frames[index].timestep) > 1e-8:
            raise ValueError('Frozen reference clock mismatch')
        return action, check

    def direct_probe(self, xy_mm, frames=15):
        self.restore()
        base = self.relation().copy()
        self.shift(xy_mm)
        shifted = self.relation().copy()
        worst = self.penetration()
        valid = worst <= .0002
        last = self.grasp_end
        if valid:
            for index in range(self.grasp_end + 1, self.grasp_end + frames + 1):
                _, check = self.step(index)
                worst = max(worst, check['max_penetration_m'])
                valid &= check['valid']
                last = index
                if not check['valid']:
                    break
        return {
            'requested_xy_mm': list(map(float, xy_mm)),
            'measured_initial_xy_mm': ((shifted - base)[:2] * 1000).tolist(),
            'retained_xy_mm': ((self.relation() - base)[:2] * 1000).tolist(),
            'max_penetration_mm': float(worst * 1000),
            'valid': bool(valid),
            'frames_advanced': last - self.grasp_end,
        }

    def force_probe(self, direction, force_n):
        self.restore()
        base = self.relation().copy()
        rotation = self.sim.data.body('right/gripper_center').xmat.reshape(3, 3)
        self.sim.data.xfrc_applied[self.plug_body, :3] = rotation @ np.r_[DIRECTIONS[direction] * force_n, 0.]
        _, first = self.step(self.grasp_end, enforce_clock=False)
        self.sim.data.xfrc_applied[self.plug_body] = 0.
        after = self.relation().copy()
        _, settled = self.step(self.grasp_end, enforce_clock=False)
        return {
            'direction': direction,
            'force_n': force_n,
            'after_force_xy_mm': ((after - base)[:2] * 1000).tolist(),
            'settled_xy_mm': ((self.relation() - base)[:2] * 1000).tolist(),
            'max_penetration_mm': float(max(first['max_penetration_m'], settled['max_penetration_m']) * 1000),
            'valid': bool(first['valid'] and settled['valid']),
        }

    def observation_episode(self, xy_mm):
        from robot_agent.full_task_observation import FullTaskProprioHistory, wrist_relative_features

        self.restore()
        base = self.relation().copy()
        self.shift(xy_mm)
        shifted = self.relation().copy()
        worst = self.penetration()
        valid = worst <= .0002
        history = FullTaskProprioHistory(15)
        vision = deque(maxlen=15)
        p0, p1, relative = [], [], []
        for index in range(self.grasp_end + 1, self.grasp_end + 16):
            action, check = self.step(index)
            worst = max(worst, check['max_penetration_m'])
            valid &= check['valid']
            qpos = self.sim.data.qpos[self.right['qpos']]
            qvel = self.sim.data.qvel[self.right['dof']]
            proprio = history.append(
                joint_position=qpos[:6], joint_velocity=qvel[:6], nominal_action=action[7:14],
                controller_target=action[7:14], gripper_width=qpos[6] + qpos[7],
                phase=self.player.frames[index].phase,
            )
            rgb = self.sim._render('right_wrist', 640, 480)
            vision.append(wrist_relative_features(rgb).vector())
            while len(vision) < 15:
                vision.appendleft(vision[0].copy())
            p0.append(proprio.copy())
            p1.append(np.r_[proprio, np.concatenate(vision)])
            relative.append(self.relation().copy())
        return {
            'summary': {
                'requested_xy_mm': list(map(float, xy_mm)),
                'measured_initial_xy_mm': ((shifted - base)[:2] * 1000).tolist(),
                'retained_xy_mm': ((self.relation() - base)[:2] * 1000).tolist(),
                'max_penetration_mm': float(worst * 1000),
                'valid': bool(valid),
            },
            'p0': np.stack(p0),
            'p1': np.stack(p1),
            'relative_position_m': np.stack(relative),
        }


def save(name, payload):
    (EXPERIMENT / name).write_text(json.dumps(payload, indent=2) + '\n')


def pilot():
    audit = SimulatorAudit()
    try:
        amplitudes = (.05, .10, .15, .18, .20, .25, .50, .75, 1., 1.25, 1.50)
        direct = []
        for amplitude in amplitudes:
            for name, vector in DIRECTIONS.items():
                row = audit.direct_probe(vector * amplitude)
                row['direction'] = name
                row['amplitude_mm'] = amplitude
                direct.append(row)
        forces = [audit.force_probe(name, force)
                  for force in (5., 10., 20., 50.) for name in DIRECTIONS]
        payload = {
            'schema_version': 1,
            'stage': 'D_PILOT',
            'validity': 'VALID',
            'hypothesis': 'INCONCLUSIVE',
            'claims_allowed': ['The direct state edit changes plug pose in the gripper frame without changing friction.'],
            'claims_not_allowed': ['The direct edit is physical slip or a measured real-world slip distribution.'],
            'program_decision': 'FREEZE_FORMAL_RANGE',
            'source': audit.source,
            'direct_state_pilot': direct,
            'contact_mediated_pilot': forces,
            'contact_mediated_status': 'PHYSICAL_SLIP_NOT_REPRODUCED',
            'candidate_cardinal_max_mm': {
                'positive_x': .18, 'negative_x': .50, 'positive_y': .18, 'negative_y': .18,
            },
            'collision_limit_mm': .2,
            'friction_changes': 0,
            'hardware_executions': 0,
            'real_data_reads': 0,
        }
        save('perturbation_pilot.json', payload)
        print(json.dumps(payload, indent=2))
    finally:
        audit.close()


def formal():
    from learn import digest

    design_path = EXPERIMENT / 'perturbation_design.json'
    design = json.loads(design_path.read_text())
    audit = SimulatorAudit()
    try:
        records = {'train': [], 'heldout': []}
        features = {
            'train': {'p0': [], 'p1': [], 'nominal_p0': [], 'nominal_p1': [],
                      'target': [], 'direction': []},
            'heldout': {'p0': [], 'p1': [], 'nominal_p0': [], 'nominal_p1': [],
                        'target': [], 'direction': []},
        }
        for split in records:
            for name, direction in DIRECTIONS.items():
                maximum = design['cardinal_max_amplitude_mm'][name]
                for fraction in design[split + '_fractions']:
                    target = direction * maximum * fraction
                    episode = audit.observation_episode(target)
                    nominal = audit.observation_episode([0., 0.])
                    episode_id = f'{split}-{name}-{fraction:.2f}'
                    records[split].append({'episode_id': episode_id, 'direction': name,
                                           'fraction': fraction, **episode['summary'],
                                           'paired_nominal_episode_id': episode_id + '-nominal',
                                           'paired_nominal': nominal['summary']})
                    for mode in ('p0', 'p1'):
                        features[split][mode].append(episode[mode])
                        features[split]['nominal_' + mode].append(nominal[mode])
                    features[split]['target'].append(
                        (episode['relative_position_m'] - nominal['relative_position_m'])[:, :2] * 1000)
                    features[split]['direction'].append(name)
        delay_metrics = {'p0': [], 'p1': []}
        train_target = np.stack(features['train']['target'])
        test_target = np.stack(features['heldout']['target'])
        final_predictions = {}
        matrix_hashes = {}
        for mode in ('p0', 'p1'):
            train = np.stack(features['train'][mode])
            test = np.stack(features['heldout'][mode])
            train_nominal = np.stack(features['train']['nominal_' + mode])
            test_nominal = np.stack(features['heldout']['nominal_' + mode])
            matrix_hashes[mode] = {
                'train_perturbed': array_hash(train), 'train_nominal': array_hash(train_nominal),
                'heldout_perturbed': array_hash(test), 'heldout_nominal': array_hash(test_nominal),
            }
            for delay in range(15):
                current_directions = [cardinal_direction(xy) for xy in test_target[:, delay]]
                metric = evaluate_xy(
                    train[:, delay], train_nominal[:, delay], train_target[:, delay],
                    test[:, delay], test_nominal[:, delay], test_target[:, delay],
                    current_directions, features['heldout']['direction'])
                metric['delay_frames'] = delay + 1
                if delay == 14:
                    final_predictions[mode] = [
                        {'episode_id': record['episode_id'],
                         'true_current_xy_mm': test_target[row, delay].tolist(),
                         'true_current_direction': current_directions[row],
                         'treatment_direction': features['heldout']['direction'][row],
                         'predicted_xy_mm': metric['predicted_xy_mm'][row],
                         'paired_nominal_predicted_xy_mm': metric['nominal_predicted_xy_mm'][row],
                         'predicted_direction': metric['predicted_directions'][row]}
                        for row, record in enumerate(records['heldout'])
                    ]
                metric.pop('predicted_xy_mm')
                metric.pop('nominal_predicted_xy_mm')
                metric.pop('predicted_directions')
                delay_metrics[mode].append(metric)
        threshold = design['observability_thresholds']
        summaries = {}
        for mode in ('p0', 'p1'):
            final = delay_metrics[mode][-1]
            passed = (final['detection_auroc'] >= threshold['detection_auroc_min']
                      and final['direction_accuracy'] >= threshold['direction_accuracy_min'])
            first = next((row['delay_frames'] for row in delay_metrics[mode]
                          if row['detection_auroc'] >= threshold['detection_auroc_min']
                          and row['direction_accuracy'] >= threshold['direction_accuracy_min']), None)
            summaries[mode] = {**final, 'passed': passed, 'first_passing_delay_frames': first}
        p0_pass = summaries['p0']['passed']
        all_valid = all(row['valid'] and row['paired_nominal']['valid']
                        for split in records.values() for row in split)
        payload = {
            'schema_version': 1,
            'audit_revision': 2,
            'supersedes': {
                'commit': '9cdb107093511871760738e18e10d2dc9f5371de',
                'artifact_sha256': '1d8ac65c3c9e962f6dd28261568a6ba9d3d533faf2205a98408e9ff66f65b39b',
                'reason': 'The first audit subtracted one shared nominal branch from classifier inputs and labeled only the initial treatment direction.',
            },
            'stage': 'C',
            'validity': 'VALID' if all_valid else 'INVALID',
            'hypothesis': 'SUPPORTED' if p0_pass else 'NOT_SUPPORTED',
            'claims_allowed': ([
                'P0 distinguishes the frozen formal perturbations and their direction in held-out simulation Episodes.'
            ] if p0_pass else [
                'P0 distinguishes perturbation presence in the deterministic held-out simulation pairs.',
            ]),
            'claims_not_allowed': [
                'The audit establishes robustness to real sensor noise or real robot transfer.',
                'P1 is deployable before simulator-to-real right-wrist viewpoint correspondence is verified.',
            ] + ([] if p0_pass else [
                'P0 or P1 identifies all four controlled perturbation directions at the required accuracy.',
            ]),
            'program_decision': 'CONTINUE' if all_valid and p0_pass else 'HOLD',
            'selected_mode': 'P0' if all_valid and p0_pass else None,
            'source': audit.source,
            'perturbation_design': str(design_path.relative_to(ROOT)),
            'perturbation_design_sha256': digest(design_path),
            'split_unit': 'paired simulated audit Episodes; disjoint amplitude fractions; one raw branch sample per Episode at each delay',
            'train_pair_count': len(records['train']),
            'heldout_pair_count': len(records['heldout']),
            'train_episode_count': 2 * len(records['train']),
            'heldout_episode_count': 2 * len(records['heldout']),
            'thresholds': threshold,
            'direction_score_definition': 'Minimum of balanced initial-treatment direction accuracy and macro recall for paired current-error direction; both must support a four-way claim.',
            'P0': {**summaries['p0'], 'deployable': p0_pass, 'sensor_interface_deployable': True,
                   'inputs': 'q, dq, frozen nominal action, command-feedback residual, gripper width, phase/history'},
            'P1': {**summaries['p1'], 'deployable': False,
                   'inputs': 'P0 plus RGB-only yellow-housing-to-finger relative XY, visibility and confidence',
                   'deployment_blocker': 'Archived metadata says current hardware extrinsics are unverified; real-data read is denied.'},
            'delay_metrics': delay_metrics,
            'final_delay_predictions': final_predictions,
            'feature_matrix_sha256': matrix_hashes,
            'episodes': records,
            'representation': 'Raw deployable P0/P1 history; no paired nominal subtraction is applied to classifier inputs.',
            'student_privileged_inputs': 0,
            'label_only_privileged_fields': [
                'requested controlled XY state edit', 'paired current plug/gripper relative XY',
                'simulation penetration',
            ],
            'limitations': [
                'All pairs originate from the one frozen frame-97 snapshot; nominal branches are separately executed but deterministic.',
                'Metrics establish treatment sensitivity only in the frozen simulator and do not include real sensor noise.',
            ],
            'hardware_executions': 0,
            'real_data_reads': 0,
        }
        save('observability.json', payload)
        print(json.dumps(payload, indent=2))
    finally:
        audit.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--formal', action='store_true')
    args = parser.parse_args()
    if args.pilot == args.formal:
        parser.error('select exactly one of --pilot or --formal')
    pilot() if args.pilot else formal()

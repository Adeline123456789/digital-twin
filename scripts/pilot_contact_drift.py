"""Pilot CONTACT_MEDIATED_DRIFT_V1 on the frozen full-task simulation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug')
EXPERIMENT = ROOT / 'experiments/residual_fulltask_mvp'
SOURCE_ID = '20261123T090549Z_a35ab76b'
PILOT_SEED = 73101
FOCUSED_SEED = 73201
FREEZE_V1_SEED = 73301
FREEZE_V2_SEED = 73401
FREEZE_V3_SEED = 73501
FREEZE_V4_SEED = 73601
TUNING_SEED = 73701
FREEZE_SEED = 73801
CONDITION_COUNT = 36
FOCUSED_CONDITION_COUNT = 24
FREEZE_CONDITION_COUNT = 24
TUNING_CONDITION_COUNT = 24
FINAL_ANGLE_INTERVALS = ((-.28, -.268), (1.05, 1.08))


def pilot_conditions():
    """Broad exploratory sweep over the amendment's initial pilot bounds."""
    rng = np.random.default_rng(PILOT_SEED)
    phases = ('lift_middle', 'transport_early', 'transport_late')
    magnitudes = (5., 10., 15., 20., 25.)
    durations = (50., 100., 150., 200.)
    angles = rng.uniform(-np.pi, np.pi, CONDITION_COUNT)
    return [
        {
            'condition_id': f'pilot_{index:03d}',
            'seed': PILOT_SEED + index,
            'phase': phases[index % len(phases)],
            'force_n': magnitudes[(index * 2 + index // len(phases)) % len(magnitudes)],
            'duration_ms': durations[(index * 3 + index // len(magnitudes)) % len(durations)],
            'angle_rad': float(angles[index]),
        }
        for index in range(CONDITION_COUNT)
    ]


def focused_conditions():
    """Focused safety/outcome calibration after the broad sweep."""
    rng = np.random.default_rng(FOCUSED_SEED)
    return [
        {
            'condition_id': f'focused_{index:03d}',
            'seed': FOCUSED_SEED + index,
            'phase': 'lift_middle',
            'force_n': 10.,
            'duration_ms': 50.,
            'angle_rad': float(angle),
        }
        for index, angle in enumerate(rng.uniform(-np.pi, np.pi, FOCUSED_CONDITION_COUNT))
    ]


def freeze_validation_conditions():
    """Fresh samples from the candidate distribution selected by the focused wave."""
    rng = np.random.default_rng(FREEZE_SEED)
    half = FREEZE_CONDITION_COUNT // 2
    angles = np.r_[rng.uniform(*FINAL_ANGLE_INTERVALS[0], half),
                   rng.uniform(*FINAL_ANGLE_INTERVALS[1], half)]
    rng.shuffle(angles)
    return [
        {
            'condition_id': f'freeze_{index:03d}',
            'seed': FREEZE_SEED + index,
            'phase': 'lift_middle',
            'force_n': 8.,
            'duration_ms': 50.,
            'angle_rad': float(angle),
        }
        for index, angle in enumerate(angles)
    ]


def tuning_conditions():
    """Resolve the smallest impulse that keeps mixed outcomes without collisions."""
    rng = np.random.default_rng(TUNING_SEED)
    rows = []
    for force in (8., 9.):
        angles = np.r_[rng.uniform(-.35, -.2, 6), rng.uniform(.9, 1.1, 6)]
        rng.shuffle(angles)
        for angle in angles:
            rows.append({
                'condition_id': f'tune_{len(rows):03d}',
                'seed': TUNING_SEED + len(rows),
                'phase': 'lift_middle',
                'force_n': force,
                'duration_ms': 50.,
                'angle_rad': float(angle),
            })
    return rows


def conditions_for_wave(wave):
    if wave == 'broad':
        return pilot_conditions()
    if wave == 'focused':
        return focused_conditions()
    return tuning_conditions() if wave == 'tune' else freeze_validation_conditions()


def rotation_angle_deg(before, after):
    relative = np.asarray(after) @ np.asarray(before).T
    cosine = np.clip((np.trace(relative) - 1) / 2, -1., 1.)
    return float(np.degrees(np.arccos(cosine)))


class ContactDriftRunner:
    def __init__(self):
        import mujoco
        from learn import SIM, digest
        from robot_agent.nominal_reference import ReferencePlayer

        sys.path.insert(0, str(SIM / 'src'))
        from piper_sim.backends.mujoco import MujocoBackend
        from piper_sim.config import load
        from piper_sim.mixed_recording import configure_rigid_collection

        manifest = json.loads((ROOT / 'experiments/event_residual_mvp/nominal_manifest.json').read_text())
        reference_path = DATA / manifest['dataset_relative_path']
        self.player = ReferencePlayer(json.loads(reference_path.read_text()), manifest['reference_hash'])
        self.raw = DATA / 'self_evolution' / SOURCE_ID
        self.stats = DATA / 'complete_sim/succeed' / SOURCE_ID / 'proprio_stats'
        task = yaml.safe_load((self.raw / 'task.yaml').read_text())
        self.sim = MujocoBackend(dict(load(), task_scene=task), cameras=False)
        self.profile = configure_rigid_collection(self.sim)
        self.mujoco = mujoco
        self.start = max(index for index, frame in enumerate(self.player.frames) if frame.phase == 'grasp')
        self.retreat_end = max(index for index, frame in enumerate(self.player.frames) if frame.phase == 'retreat')
        self.end = len(self.player.frames) - 1
        with (self.stats / 'privileged.jsonl').open() as stream:
            privileged = next(json.loads(line) for index, line in enumerate(stream) if index == self.start)
        self.snapshot = privileged['physics_state']
        self.plug_body = self.sim.model.body('task/yellow_plug').id
        lift = [index for index, frame in enumerate(self.player.frames) if frame.phase == 'lift']
        transport = [index for index, frame in enumerate(self.player.frames) if frame.phase == 'transport']
        self.anchors = {
            'lift_middle': lift[len(lift) // 2],
            'transport_early': transport[len(transport) // 6],
            'transport_late': transport[5 * len(transport) // 6],
        }
        sim_files = [
            SIM / 'src/piper_sim/backends/mujoco.py',
            SIM / 'src/piper_sim/mixed_recording.py',
            SIM / 'src/piper_sim/tasks/recorded_scenes.py',
        ]
        sim_head = subprocess.check_output(
            ['git', '-C', str(SIM), 'rev-parse', 'HEAD'], text=True).strip()
        sim_dirty = bool(subprocess.check_output(
            ['git', '-C', str(SIM), 'status', '--porcelain'], text=True).strip())
        self.source = {
            'episode': SOURCE_ID,
            'frozen_reference_sha256': digest(reference_path),
            'model_sha256': digest(self.stats / 'model.xml'),
            'simulator_git_head': sim_head,
            'simulator_worktree_dirty': sim_dirty,
            'simulator_source_sha256': {
                str(path.relative_to(SIM)): digest(path) for path in sim_files
            },
            'grasp_established_frame': self.start,
            'retreat_end_frame': self.retreat_end,
            'episode_end_frame': self.end,
            'physics_state_mujoco_version': self.snapshot['mujoco_version'],
            'physics_state_spec': self.snapshot['spec'],
            'physics_timestep_s': float(self.sim.model.opt.timestep),
            'observation_timestep_s': float(self.sim.dt * self.sim.decimation),
            'integrator_substeps_per_action': int(self.sim.decimation * self.sim.substeps),
            'phase_anchor_frames': self.anchors,
            'force_onset': 'first integrator substep after the completed anchor frame',
            'force_frame': 'gripper-local XY transformed by the instantaneous gripper rotation every substep',
            'zero_force_pairing': 'matched branch from the same frame-97 integration state and frozen suffix actions',
        }
        self.nominal_cache = {}

    def close(self):
        self.sim.close()

    def restore(self):
        self.sim.reset()
        self.mujoco.mj_setState(
            self.sim.model, self.sim.data, np.asarray(self.snapshot['values']), self.snapshot['spec'])
        self.sim.task.hold = 0.
        self.mujoco.mj_forward(self.sim.model, self.sim.data)
        if abs(self.sim.sim_time - self.player.frames[self.start].timestep) > 1e-8:
            raise ValueError('Restored frame-97 clock mismatch')
        if not np.allclose(self.sim.control_targets(), self.player.frames[self.start].nominal_action,
                           atol=1e-10, rtol=0.):
            raise ValueError('Restored frame-97 control mismatch')

    def relation(self):
        grip = self.sim.data.body('right/gripper_center')
        plug = self.sim.data.body('task/yellow_plug')
        rotation = grip.xmat.reshape(3, 3).T
        return {
            'position_m': (rotation @ (plug.xpos - grip.xpos)).tolist(),
            'rotation_matrix': (rotation @ plug.xmat.reshape(3, 3)).tolist(),
        }

    def aperture_m(self):
        qpos = self.sim.data.qpos[self.sim.indices['right']['qpos']]
        return float(qpos[6] + qpos[7])

    def plug_contacts(self):
        contacts = []
        for contact in self.sim.data.contact:
            geoms = (int(contact.geom1), int(contact.geom2))
            bodies = tuple(int(self.sim.model.geom_bodyid[geom]) for geom in geoms)
            if self.plug_body not in bodies:
                continue
            contacts.append({
                'geom_pair': sorted(self.sim.model.geom(geom).name for geom in geoms),
                'body_pair': sorted(self.sim.model.body(body).name for body in bodies),
                'distance_m': float(contact.dist),
                'solver_active': bool(contact.efc_address >= 0),
            })
        return sorted(contacts, key=lambda row: (row['body_pair'], row['geom_pair'], row['distance_m']))

    @staticmethod
    def contact_summary(samples):
        grouped = {}
        for contact in samples:
            key = (tuple(contact['body_pair']), tuple(contact['geom_pair']))
            row = grouped.setdefault(key, {
                'body_pair': list(key[0]), 'geom_pair': list(key[1]),
                'samples': 0, 'solver_active_samples': 0, 'min_distance_m': float('inf'),
            })
            row['samples'] += 1
            row['solver_active_samples'] += int(contact['solver_active'])
            row['min_distance_m'] = min(row['min_distance_m'], contact['distance_m'])
        return sorted(grouped.values(), key=lambda row: (row['body_pair'], row['geom_pair']))

    def measurement(self):
        return {
            **self.relation(),
            'gripper_aperture_m': self.aperture_m(),
            'contact_topology': self.plug_contacts(),
        }

    @staticmethod
    def metrics_json(metrics):
        return {
            key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
            for key, value in metrics.items()
        }

    def run_branch(self, condition, apply_force):
        self.restore()
        anchor = self.anchors[condition['phase']]
        force_steps = int(round(condition['duration_ms'] / 1000 / self.sim.model.opt.timestep))
        settle_steps = int(round(.2 / self.sim.model.opt.timestep))
        requested = np.array([np.cos(condition['angle_rad']), np.sin(condition['angle_rad']), 0.])
        elapsed = None
        before = force_end = settled = None
        force_contacts, settle_contacts = [], []
        worst = 0.
        valid = True
        last = self.start
        try:
            for index in range(self.start + 1, self.end + 1):
                action = np.asarray(self.player.frames[index].nominal_action)
                self.sim.apply(self.sim.layout.command(action, self.sim.sim_time, index))
                completed_interval = True
                for _ in range(self.sim.decimation * self.sim.substeps):
                    in_force_window = elapsed is not None and elapsed < force_steps
                    self.sim.data.xfrc_applied[self.plug_body] = 0.
                    if apply_force and in_force_window:
                        rotation = self.sim.data.body('right/gripper_center').xmat.reshape(3, 3)
                        self.sim.data.xfrc_applied[self.plug_body, :3] = (
                            rotation @ requested * condition['force_n'])
                    self.mujoco.mj_step(self.sim.model, self.sim.data)
                    self.mujoco.mj_forward(self.sim.model, self.sim.data)
                    if self.sim.data.ncon:
                        worst = max(worst, -min(contact.dist for contact in self.sim.data.contact))
                    if elapsed is not None:
                        if elapsed < force_steps:
                            force_contacts.extend(self.plug_contacts())
                        elif elapsed < force_steps + settle_steps:
                            settle_contacts.extend(self.plug_contacts())
                        elapsed += 1
                        if elapsed == force_steps:
                            force_end = self.measurement()
                        elif elapsed == force_steps + settle_steps:
                            settled = self.measurement()
                    if (worst > .0002 or not np.isfinite(self.sim.data.qpos).all()
                            or not np.isfinite(self.sim.data.qvel).all()):
                        valid = False
                        completed_interval = False
                        break
                self.sim.data.xfrc_applied[self.plug_body] = 0.
                if not completed_interval:
                    break
                self.sim._images = None
                self.sim.task.metrics(
                    self.sim.task_adapter.task_poses(self.sim.data, self.sim.task),
                    self.sim.dt * self.sim.decimation)
                last = index
                if abs(self.sim.sim_time - self.player.frames[index].timestep) > 1e-8:
                    raise ValueError('Frozen reference clock mismatch')
                if index == anchor:
                    before = self.measurement()
                    elapsed = 0
        finally:
            self.sim.data.xfrc_applied[self.plug_body] = 0.
        final_metrics = self.sim.task.metrics(
            self.sim.task_adapter.task_poses(self.sim.data, self.sim.task))
        force_summary = self.contact_summary(force_contacts)
        settle_summary = self.contact_summary(settle_contacts)
        finger_names = {'right/link7', 'right/link8'}
        finger_contact = any(
            finger_names.intersection(row['body_pair']) and row['solver_active_samples']
            for row in settle_summary)
        return {
            'branch': 'forced' if apply_force else 'matched_zero_force',
            'valid': bool(valid and force_end is not None and settled is not None),
            'max_penetration_mm': float(worst * 1000),
            'last_frame': last,
            'last_phase': self.player.frames[last].phase,
            'task_success': bool(valid and last == self.end and final_metrics['success']),
            'before': before,
            'force_end': force_end,
            'settled_200ms': settled,
            'force_window_contact_summary': force_summary,
            'settle_window_contact_summary': settle_summary,
            'plug_finger_contact_during_settle': bool(finger_contact),
            'final_metrics': self.metrics_json(final_metrics),
        }

    @staticmethod
    def paired_drift(nominal, forced):
        result = {}
        for name in ('force_end', 'settled_200ms'):
            zero = nominal[name]
            disturbed = forced[name]
            if zero is None or disturbed is None:
                result[name] = None
                continue
            translation = (np.asarray(disturbed['position_m']) - np.asarray(zero['position_m'])) * 1000
            result[name] = {
                'translation_mm': translation.tolist(),
                'translation_norm_mm': float(np.linalg.norm(translation)),
                'rotation_deg': rotation_angle_deg(
                    zero['rotation_matrix'], disturbed['rotation_matrix']),
                'gripper_aperture_delta_mm': float(
                    (disturbed['gripper_aperture_m'] - zero['gripper_aperture_m']) * 1000),
            }
        return result

    def run(self, condition):
        pair_key = (condition['phase'], condition['duration_ms'])
        if pair_key not in self.nominal_cache:
            self.nominal_cache[pair_key] = self.run_branch(condition, apply_force=False)
        nominal = self.nominal_cache[pair_key]
        forced = self.run_branch(condition, apply_force=True)
        semantics_intact = forced['plug_finger_contact_during_settle']
        valid = nominal['valid'] and nominal['task_success'] and forced['valid'] and semantics_intact
        return {
            **condition,
            'anchor_frame': self.anchors[condition['phase']],
            'duration_substeps': int(round(
                condition['duration_ms'] / 1000 / self.sim.model.opt.timestep)),
            'valid': bool(valid),
            'invalid_reason': None if valid else (
                'MATCHED_ZERO_FORCE_INVALID' if not nominal['valid'] or not nominal['task_success'] else
                'COLLISION_BOUND' if not forced['valid'] else 'TASK_SEMANTICS_DESTROYED'),
            'task_success': bool(valid and forced['task_success']),
            'actual_drift_vs_matched_zero_force': self.paired_drift(nominal, forced),
            'matched_zero_force': nominal,
            'disturbed_frozen_nominal': forced,
            'prohibited_changes': {
                'friction': 0, 'mass': 0, 'contact_solver': 0,
                'success_detector': 0, 'frozen_nominal': 0,
            },
            'hardware_executions': 0,
            'real_data_reads': 0,
        }


def run_worker(worker, workers, output, wave):
    output.mkdir(parents=True, exist_ok=True)
    runner = ContactDriftRunner()
    try:
        for index, condition in enumerate(conditions_for_wave(wave)):
            if index % workers != worker:
                continue
            path = output / f"{condition['condition_id']}.json"
            result = runner.run(condition)
            path.write_text(json.dumps(result, indent=2) + '\n')
            drift = result['actual_drift_vs_matched_zero_force']['settled_200ms']
            print('PILOT_CONDITION_COMPLETE', json.dumps({
                'condition_id': condition['condition_id'], 'valid': result['valid'],
                'task_success': result['task_success'],
                'drift_mm': None if drift is None else drift['translation_norm_mm'],
                'invalid_reason': result['invalid_reason'],
            }), flush=True)
    finally:
        runner.close()


def aggregate(output, wave):
    from learn import digest

    conditions = conditions_for_wave(wave)
    all_rows = [json.loads((output / f"{condition['condition_id']}.json").read_text())
                for condition in conditions]
    excluded = []
    rows = all_rows
    if wave == 'freeze':
        rows = [row for row in all_rows if any(
            low <= row['angle_rad'] <= high for low, high in FINAL_ANGLE_INTERVALS)]
        excluded = [
            {'condition_id': row['condition_id'], 'angle_rad': row['angle_rad'],
             'reason': row['invalid_reason'] or 'OUTSIDE_FINAL_SAFETY_INTERVAL'}
            for row in all_rows if row not in rows
        ]
    valid = [row for row in rows if row['valid']]
    summary = {
        'conditions': len(rows),
        'valid_conditions': len(valid),
        'invalid_reasons': dict(Counter(
            row['invalid_reason'] for row in rows if not row['valid'])),
        'successes': sum(row['task_success'] for row in valid),
        'failures': sum(not row['task_success'] for row in valid),
        'frozen_nominal_success_rate': (
            sum(row['task_success'] for row in valid) / len(valid)) if valid else None,
        'outcomes_by_phase': {
            phase: dict(Counter('success' if row['task_success'] else 'failure'
                                for row in valid if row['phase'] == phase))
            for phase in ('lift_middle', 'transport_early', 'transport_late')
        },
    }
    source_runner = ContactDriftRunner()
    try:
        source = source_runner.source
    finally:
        source_runner.close()
    calibration_history = None
    if wave == 'freeze':
        focused_paths = sorted((ROOT / '.local/residual_fulltask_mvp/contact_drift_pilot_focused_v2').glob('*.json'))
        broad_paths = sorted((ROOT / '.local/residual_fulltask_mvp/contact_drift_pilot_wave1').glob('*.json'))
        freeze_v1_paths = sorted((ROOT / '.local/residual_fulltask_mvp/contact_drift_pilot_freeze_v1').glob('*.json'))
        freeze_v2_paths = sorted((ROOT / '.local/residual_fulltask_mvp/contact_drift_pilot_freeze_v2').glob('*.json'))
        freeze_v3_paths = sorted((ROOT / '.local/residual_fulltask_mvp/contact_drift_pilot_freeze_v3').glob('*.json'))
        freeze_v4_paths = sorted((ROOT / '.local/residual_fulltask_mvp/contact_drift_pilot_freeze_v4').glob('*.json'))
        tuning_paths = sorted((ROOT / '.local/residual_fulltask_mvp/contact_drift_pilot_tune_v1').glob('*.json'))
        focused = [json.loads(path.read_text()) for path in focused_paths]
        broad = [json.loads(path.read_text()) for path in broad_paths]
        freeze_v1 = [json.loads(path.read_text()) for path in freeze_v1_paths]
        freeze_v2 = [json.loads(path.read_text()) for path in freeze_v2_paths]
        freeze_v3 = [json.loads(path.read_text()) for path in freeze_v3_paths]
        freeze_v4 = [json.loads(path.read_text()) for path in freeze_v4_paths]
        tuning = [json.loads(path.read_text()) for path in tuning_paths]

        def compact(rows):
            accepted = [row for row in rows if row['valid']]
            return {
                'conditions': len(rows),
                'valid_conditions': len(accepted),
                'successes': sum(row['task_success'] for row in accepted),
                'failures': sum(not row['task_success'] for row in accepted),
                'success_rate': (sum(row['task_success'] for row in accepted) / len(accepted)
                                 if accepted else None),
            }

        calibration_history = {
            'broad_exploration': {
                **compact(broad),
                'seed': PILOT_SEED,
                'domain': '5-25 N, 50-200 ms, three requested phases, continuous random angle',
                'status': 'EXPLORATORY_ONLY',
                'limit': 'Stopped at retreat and measured within-branch settling; used only to choose the corrected focused wave.',
            },
            'corrected_focused_wave': {
                **compact(focused),
                'seed': FOCUSED_SEED,
                'domain': 'lift-middle, 10 N, 50 ms, full continuous angle',
                'status': 'CALIBRATION',
                'selection': 'The 0.95-1.35 rad interval contained valid successes and failures with penetration margin.',
            },
            'fresh_freeze_validation': {
                **summary,
                'seed': FREEZE_SEED,
                'domain': 'lift-middle, 8 N, 50 ms, equal mixture of uniform angles -0.28--0.268 and 1.05-1.08 rad',
                'status': 'FORMAL_FREEZE_EVIDENCE',
                'safety_rejection': excluded,
            },
            'first_candidate_validation': {
                **compact(freeze_v1),
                'seed': FREEZE_V1_SEED,
                'domain': 'lift-middle, 10 N, 50 ms, uniform angle 0.95-1.35 rad',
                'status': 'REFINED',
                'reason': 'All conditions were valid but the 87.5% success rate was weaker than the preferred mixed challenge.',
            },
            'second_candidate_validation': {
                **compact(freeze_v2),
                'seed': FREEZE_V2_SEED,
                'domain': 'lift-middle, 10 N, 50 ms, uniform angle 0.98-1.05 rad',
                'status': 'ABORTED_AFTER_INVALID_SAMPLE',
                'reason': 'A near-identical angle crossed the collision limit, exposing an unsafe contact bifurcation.',
            },
            'third_candidate_validation': {
                **compact(freeze_v3),
                'seed': FREEZE_V3_SEED,
                'domain': 'lift-middle, 10 N, 50 ms, equal mixture of uniform angles 0.978-0.998 and 1.12-1.20 rad',
                'status': 'ABORTED_AFTER_INVALID_SAMPLE',
                'reason': 'The positive failure interval still contained a narrow collision island.',
            },
            'fourth_candidate_validation': {
                **compact(freeze_v4),
                'seed': FREEZE_V4_SEED,
                'domain': 'lift-middle, 10 N, 50 ms, equal mixture of uniform angles -0.32--0.28 and 1.25-1.33 rad',
                'status': 'ABORTED_AFTER_INVALID_SAMPLE',
                'reason': 'The negative failure interval also contained a collision bifurcation; force magnitude required refinement.',
            },
            'magnitude_refinement': {
                'seed': TUNING_SEED,
                'domain': 'lift-middle, 50 ms, 8 N versus 9 N in two sensitive angle bands',
                'force_8n': compact([row for row in tuning if row['force_n'] == 8.]),
                'force_9n': compact([row for row in tuning if row['force_n'] == 9.]),
                'selection': '8 N retained valid failures and a stable success band; 9 N negative-direction samples crossed the collision limit.',
            },
        }
    ready = (wave == 'freeze' and len(valid) == len(rows)
             and .2 <= summary['frozen_nominal_success_rate'] <= .8)
    payload = {
        'schema_version': 2,
        'benchmark': 'CONTACT_MEDIATED_DRIFT_V1',
        'stage': 'PILOT',
        'wave': wave,
        'validity': 'VALID' if valid else 'INVALID',
        'hypothesis': 'SUPPORTED' if summary['successes'] and summary['failures'] else 'INCONCLUSIVE',
        'claims_allowed': [
            'Unchanged-physics external disturbances produce the recorded paired plug-in-gripper drift.',
        ],
        'claims_not_allowed': [
            'Requested force direction equals actual drift direction.',
            'Pilot bounds estimate a real-world disturbance distribution.',
        ],
        'program_decision': 'FREEZE_CHALLENGE_DISTRIBUTION' if ready else 'REFINE_PILOT',
        'pilot_seed': {
            'broad': PILOT_SEED, 'focused': FOCUSED_SEED,
            'tune': TUNING_SEED, 'freeze': FREEZE_SEED,
        }[wave],
        'source': source,
        'script_sha256': digest(Path(__file__)),
        'summary': summary,
        'calibration_history': calibration_history,
        'conditions': rows,
        'excluded_calibration_conditions': excluded,
        'permissions': {
            'hardware_executions': 0, 'real_data_reads': 0, 'real_training': 0,
            'runtime_llm_calls': 0, 'automatic_twin_mutations': 0,
        },
    }
    (EXPERIMENT / 'contact_drift_pilot.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload['summary'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave', choices=('broad', 'focused', 'tune', 'freeze'), default='focused')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    output = args.output or ROOT / f'.local/residual_fulltask_mvp/contact_drift_pilot_{args.wave}'
    if args.aggregate:
        aggregate(output, args.wave)
    elif args.worker is not None:
        run_worker(args.worker, args.workers, output, args.wave)
    else:
        parser.error('select --worker or --aggregate')


if __name__ == '__main__':
    main()

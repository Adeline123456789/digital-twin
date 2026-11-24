"""Pilot CONTACT_MEDIATED_DRIFT_V1 on the frozen full-task simulation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug')
EXPERIMENT = ROOT / 'experiments/residual_fulltask_mvp'
SOURCE_ID = '20261123T090549Z_a35ab76b'
PILOT_SEED = 73101
CONDITION_COUNT = 36


def pilot_conditions():
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
        self.end = max(index for index, frame in enumerate(self.player.frames) if frame.phase == 'retreat')
        privileged = json.loads(next(line for index, line in enumerate((self.stats / 'privileged.jsonl').open())
                                     if index == self.start))
        self.snapshot = privileged['physics_state']
        self.plug_body = self.sim.model.body('task/yellow_plug').id
        lift = [index for index, frame in enumerate(self.player.frames) if frame.phase == 'lift']
        transport = [index for index, frame in enumerate(self.player.frames) if frame.phase == 'transport']
        self.anchors = {
            'lift_middle': lift[len(lift) // 2],
            'transport_early': transport[len(transport) // 6],
            'transport_late': transport[5 * len(transport) // 6],
        }
        self.source = {
            'episode': SOURCE_ID,
            'frozen_reference_sha256': digest(reference_path),
            'model_sha256': digest(self.stats / 'model.xml'),
            'grasp_established_frame': self.start,
            'retreat_end_frame': self.end,
            'physics_state_mujoco_version': self.snapshot['mujoco_version'],
            'physics_state_spec': self.snapshot['spec'],
            'physics_timestep_s': float(self.sim.model.opt.timestep),
            'observation_timestep_s': float(self.sim.dt * self.sim.decimation),
            'phase_anchor_frames': self.anchors,
        }

    def close(self):
        self.sim.close()

    def restore(self):
        self.sim.reset()
        self.mujoco.mj_setState(self.sim.model, self.sim.data,
                                np.asarray(self.snapshot['values']), self.snapshot['spec'])
        self.mujoco.mj_forward(self.sim.model, self.sim.data)

    def relation(self):
        grip = self.sim.data.body('right/gripper_center')
        plug = self.sim.data.body('task/yellow_plug')
        rotation = grip.xmat.reshape(3, 3).T
        return {
            'position_m': rotation @ (plug.xpos - grip.xpos),
            'rotation': rotation @ plug.xmat.reshape(3, 3),
        }

    def aperture_m(self):
        qpos = self.sim.data.qpos[self.sim.indices['right']['qpos']]
        return float(qpos[6] + qpos[7])

    def plug_contacts(self):
        pairs = []
        for contact in self.sim.data.contact:
            bodies = [self.sim.model.geom_bodyid[contact.geom1], self.sim.model.geom_bodyid[contact.geom2]]
            if self.plug_body not in bodies:
                continue
            pairs.append(tuple(sorted(self.sim.model.body(body).name for body in bodies)))
        return pairs

    def snapshot_measurement(self, before):
        current = self.relation()
        return {
            'translation_mm': ((current['position_m'] - before['position_m']) * 1000).tolist(),
            'translation_norm_mm': float(np.linalg.norm(current['position_m'] - before['position_m']) * 1000),
            'rotation_deg': rotation_angle_deg(before['rotation'], current['rotation']),
            'gripper_aperture_m': self.aperture_m(),
            'contact_topology': [list(pair) for pair in sorted(set(self.plug_contacts()))],
        }

    def run(self, condition):
        self.restore()
        anchor = self.anchors[condition['phase']]
        force_steps = int(round(condition['duration_ms'] / 1000 / self.sim.model.opt.timestep))
        settle_steps = int(round(.2 / self.sim.model.opt.timestep))
        remaining_force = force_steps
        remaining_settle = None
        before = force_end = settled = None
        before_aperture = None
        before_contacts, during_contacts, settle_contacts = [], [], []
        worst = 0.
        valid = True
        last = self.start
        requested = np.array([np.cos(condition['angle_rad']), np.sin(condition['angle_rad']), 0.])
        for index in range(self.start + 1, self.end + 1):
            action = np.asarray(self.player.frames[index].nominal_action)
            self.sim.apply(self.sim.layout.command(action, self.sim.sim_time, index))
            if index == anchor:
                before = self.relation()
                before_aperture = self.aperture_m()
                before_contacts = self.plug_contacts()
            for _ in range(self.sim.decimation * self.sim.substeps):
                forcing = before is not None and remaining_force > 0
                if forcing:
                    rotation = self.sim.data.body('right/gripper_center').xmat.reshape(3, 3)
                    self.sim.data.xfrc_applied[self.plug_body, :3] = (
                        rotation @ requested * condition['force_n'])
                else:
                    self.sim.data.xfrc_applied[self.plug_body] = 0.
                self.mujoco.mj_step(self.sim.model, self.sim.data)
                self.mujoco.mj_forward(self.sim.model, self.sim.data)
                if self.sim.data.ncon:
                    worst = max(worst, -min(contact.dist for contact in self.sim.data.contact))
                if forcing:
                    during_contacts.extend(self.plug_contacts())
                    remaining_force -= 1
                    if remaining_force == 0:
                        self.sim.data.xfrc_applied[self.plug_body] = 0.
                        force_end = self.snapshot_measurement(before)
                        remaining_settle = settle_steps
                elif remaining_settle is not None and remaining_settle > 0:
                    settle_contacts.extend(self.plug_contacts())
                    remaining_settle -= 1
                    if remaining_settle == 0:
                        settled = self.snapshot_measurement(before)
                if worst > .0002 or not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
                    valid = False
                    break
            self.sim.data.xfrc_applied[self.plug_body] = 0.
            self.sim._images = None
            self.sim.task.metrics(self.sim.task_adapter.task_poses(self.sim.data, self.sim.task),
                                  self.sim.dt * self.sim.decimation)
            last = index
            if valid and abs(self.sim.sim_time - self.player.frames[index].timestep) > 1e-8:
                raise ValueError('Frozen reference clock mismatch')
            if not valid:
                break
        final_metrics = self.sim.task.metrics(self.sim.task_adapter.task_poses(self.sim.data, self.sim.task))
        if force_end is not None:
            force_end['contact_topology'] = [list(pair) for pair in sorted(set(during_contacts))]
        if settled is not None:
            settled['contact_topology'] = [list(pair) for pair in sorted(set(settle_contacts))]
        finger_names = {'right/link7', 'right/link8'}
        settled_pairs = set(tuple(pair) for pair in (settled or {}).get('contact_topology', []))
        finger_contact = any(finger_names.intersection(pair) for pair in settled_pairs)
        return {
            **condition,
            'anchor_frame': anchor,
            'duration_substeps': force_steps,
            'valid': bool(valid and force_end is not None and settled is not None),
            'max_penetration_mm': float(worst * 1000),
            'last_frame': last,
            'last_phase': self.player.frames[last].phase,
            'task_success': bool(valid and last == self.end and final_metrics['success']),
            'before': {
                'gripper_aperture_m': before_aperture,
                'contact_topology': [list(pair) for pair in sorted(set(before_contacts))],
            },
            'force_end': force_end,
            'settled_200ms': settled,
            'plug_finger_contact_after_settle': finger_contact,
            'final_metrics': {
                key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
                for key, value in final_metrics.items()
            },
            'prohibited_changes': {
                'friction': 0, 'mass': 0, 'contact_solver': 0,
                'success_detector': 0, 'frozen_nominal': 0,
            },
            'hardware_executions': 0,
            'real_data_reads': 0,
        }


def run_worker(worker, workers, output):
    output.mkdir(parents=True, exist_ok=True)
    runner = ContactDriftRunner()
    try:
        for index, condition in enumerate(pilot_conditions()):
            if index % workers != worker:
                continue
            path = output / f"{condition['condition_id']}.json"
            result = runner.run(condition)
            path.write_text(json.dumps(result, indent=2) + '\n')
            print('PILOT_CONDITION_COMPLETE', json.dumps({
                'condition_id': condition['condition_id'], 'valid': result['valid'],
                'task_success': result['task_success'],
                'drift_mm': (result['settled_200ms'] or {}).get('translation_norm_mm'),
            }), flush=True)
    finally:
        runner.close()


def aggregate(output):
    from learn import digest

    conditions = pilot_conditions()
    rows = [json.loads((output / f"{condition['condition_id']}.json").read_text())
            for condition in conditions]
    valid = [row for row in rows if row['valid']]
    summary = {
        'conditions': len(rows),
        'valid_conditions': len(valid),
        'successes': sum(row['task_success'] for row in valid),
        'failures': sum(not row['task_success'] for row in valid),
        'nominal_success_rate': (sum(row['task_success'] for row in valid) / len(valid)) if valid else None,
        'finger_contact_after_settle': sum(row['plug_finger_contact_after_settle'] for row in valid),
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
    payload = {
        'schema_version': 1,
        'benchmark': 'CONTACT_MEDIATED_DRIFT_V1',
        'stage': 'PILOT',
        'validity': 'VALID' if valid else 'INVALID',
        'hypothesis': 'SUPPORTED' if summary['successes'] and summary['failures'] else 'INCONCLUSIVE',
        'claims_allowed': [
            'Unchanged-physics external disturbances produce the recorded actual plug-in-gripper drift.',
        ],
        'claims_not_allowed': [
            'Requested force direction equals actual drift direction.',
            'Pilot bounds estimate a real-world disturbance distribution.',
        ],
        'program_decision': 'FREEZE_CHALLENGE_DISTRIBUTION' if (
            valid and .2 <= summary['nominal_success_rate'] <= .8) else 'REFINE_PILOT',
        'pilot_seed': PILOT_SEED,
        'source': source,
        'script_sha256': digest(Path(__file__)),
        'summary': summary,
        'conditions': rows,
        'permissions': {
            'hardware_executions': 0, 'real_data_reads': 0, 'real_training': 0,
            'runtime_llm_calls': 0, 'automatic_twin_mutations': 0,
        },
    }
    (EXPERIMENT / 'contact_drift_pilot.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload['summary'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT / '.local/residual_fulltask_mvp/contact_drift_pilot')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.output)
    elif args.worker is not None:
        run_worker(args.worker, args.workers, args.output)
    else:
        parser.error('select --worker or --aggregate')


if __name__ == '__main__':
    main()

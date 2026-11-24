"""Run the frozen nominal through the formal post-grasp perturbation boundaries."""
import json
from pathlib import Path

import numpy as np

from audit_post_grasp_observability import DIRECTIONS, EXPERIMENT, ROOT, SimulatorAudit


def json_metrics(values):
    return {key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
            for key, value in values.items()}


def rollout(audit, xy_mm):
    audit.restore()
    base = audit.relation().copy()
    audit.shift(xy_mm)
    shifted = audit.relation().copy()
    worst = audit.penetration()
    valid = worst <= .0002
    phase_end = {}
    end = max(i for i, frame in enumerate(audit.player.frames) if frame.phase == 'retreat')
    last = audit.grasp_end
    for index in range(audit.grasp_end + 1, end + 1):
        _, check = audit.step(index)
        worst = max(worst, check['max_penetration_m'])
        valid &= check['valid']
        last = index
        phase = audit.player.frames[index].phase
        if index == end or audit.player.frames[index + 1].phase != phase:
            phase_end[phase] = {
                'plug_in_gripper_frame_m': audit.relation().tolist(),
                'task_metrics': json_metrics(audit.sim.task.metrics(
                    audit.sim.task_adapter.task_poses(audit.sim.data, audit.sim.task))),
            }
        if not check['valid']:
            break
    metrics = audit.sim.task.metrics(audit.sim.task_adapter.task_poses(audit.sim.data, audit.sim.task))
    return {
        'requested_xy_mm': list(map(float, xy_mm)),
        'measured_initial_xy_mm': ((shifted - base)[:2] * 1000).tolist(),
        'valid': bool(valid),
        'max_penetration_mm': float(worst * 1000),
        'last_frame': last,
        'last_phase': audit.player.frames[last].phase,
        'task_success': bool(valid and metrics['success']),
        'final_metrics': json_metrics(metrics),
        'phase_end': phase_end,
    }


def main():
    from learn import digest

    design_path = EXPERIMENT / 'perturbation_design.json'
    pilot_path = EXPERIMENT / 'perturbation_pilot.json'
    design = json.loads(design_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    audit = SimulatorAudit()
    try:
        nominal = rollout(audit, [0., 0.])
        episodes = []
        for name, direction in DIRECTIONS.items():
            target = direction * design['cardinal_max_amplitude_mm'][name]
            row = rollout(audit, target)
            row['direction'] = name
            for phase, evidence in row['phase_end'].items():
                if phase in nominal['phase_end']:
                    evidence['paired_relative_position_delta_mm'] = (
                        (np.asarray(evidence['plug_in_gripper_frame_m'])
                         - nominal['phase_end'][phase]['plug_in_gripper_frame_m']) * 1000
                    ).tolist()
            episodes.append(row)
        all_valid = nominal['valid'] and all(row['valid'] for row in episodes)
        failures = sum(not row['task_success'] for row in episodes)
        payload = {
            'schema_version': 1,
            'stage': 'D',
            'validity': 'VALID' if all_valid else 'INVALID',
            'hypothesis': 'SUPPORTED' if all_valid else 'INCONCLUSIVE',
            'claims_allowed': [
                'The frozen controlled cardinal family changes post-grasp plug state without friction changes and stays within the collision bound.',
                f'{failures} of {len(episodes)} boundary perturbations caused the frozen nominal task to fail.',
            ] if all_valid else [],
            'claims_not_allowed': [
                'The controlled qpos edit is contact-mediated physical slip.',
                'The perturbation amplitudes estimate a real-world slip distribution.',
                'A recovery benefit is established before residual training and held-out evaluation.',
                'The frozen family causes nominal task failures.' if not failures else
                'Every valid perturbation direction causes nominal task failure.',
            ],
            'program_decision': 'CONTINUE' if all_valid else 'HOLD',
            'program_decision_reason': (
                'The controlled perturbation family is valid, satisfying the Stage D hard prerequisite. '
                'Zero boundary failures limit the later recovery claim but do not invalidate the family.'
                if all_valid else 'At least one frozen boundary condition is physically invalid.'
            ),
            'nominal_failure_hypothesis': 'SUPPORTED' if failures else 'NOT_SUPPORTED',
            'perturbation_type': design['perturbation_type'],
            'coordinate_frame': design['coordinate_frame'],
            'design': str(design_path.relative_to(ROOT)),
            'design_sha256': digest(design_path),
            'pilot': str(pilot_path.relative_to(ROOT)),
            'pilot_sha256': digest(pilot_path),
            'contact_mediated_status': pilot['contact_mediated_status'],
            'friction_changes': 0,
            'nominal': nominal,
            'boundary_episodes': episodes,
            'boundary_failure_count': failures,
            'hardware_executions': 0,
            'real_data_reads': 0,
            'student_privileged_inputs': 0,
        }
        (EXPERIMENT / 'perturbation_audit.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload, indent=2))
    finally:
        audit.close()


if __name__ == '__main__':
    main()

"""Audit and export the existing complete frozen dual-arm simulation reference."""
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug')
SOURCE_ID = '20261123T090549Z_a35ab76b'
REQUIRED_TASK_SEQUENCE = ('APPROACH', 'GRASP', 'LIFT', 'TRANSPORT', 'ALIGN', 'INSERT', 'RELEASE', 'RETREAT')
TASK_PHASE = {
    'descend': 'APPROACH',
    'grasp': 'GRASP',
    'lift': 'LIFT',
    'transport': 'TRANSPORT',
    'align': 'ALIGN',
    'insert': 'INSERT',
    'release': 'RELEASE',
    'retreat': 'RETREAT',
    'fold': 'POST_RETREAT',
}


def runs(values):
    return [value for index, value in enumerate(values) if index == 0 or value != values[index - 1]]


def task_sequence(phases):
    sequence = runs([TASK_PHASE[phase] for phase in phases])
    if tuple(sequence[:len(REQUIRED_TASK_SEQUENCE)]) != REQUIRED_TASK_SEQUENCE:
        raise ValueError('Frozen reference does not cover the required full-task phase order')
    return sequence


def export_rows(frames):
    rows = []
    for index, frame in enumerate(frames):
        action = list(frame.nominal_action)
        rows.append({
            'index': index,
            'timestep_s': frame.timestep,
            'native_phase': frame.phase,
            'task_phase': TASK_PHASE[frame.phase],
            'left_joint_target_rad': action[:6],
            'left_gripper_target_m': action[6],
            'right_joint_target_rad': action[7:13],
            'right_gripper_target_m': action[13],
            'nominal_action': action,
        })
    return rows


def main():
    sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
    from learn import digest, save
    from episode_format import validate
    from robot_agent.nominal_reference import ReferencePlayer

    experiment = ROOT / 'experiments/residual_fulltask_mvp'
    prior = ROOT / 'experiments/event_residual_mvp'
    nominal_manifest_path = prior / 'nominal_manifest.json'
    nominal_manifest = json.loads(nominal_manifest_path.read_text())
    if nominal_manifest['reference_episode_id'] != SOURCE_ID:
        raise ValueError('Unexpected frozen source Episode')

    reference_path = DATA / nominal_manifest['dataset_relative_path']
    if digest(reference_path) != nominal_manifest['sha256']:
        raise ValueError('Frozen reference bytes changed')
    player = ReferencePlayer(json.loads(reference_path.read_text()), nominal_manifest['reference_hash'])

    trial = DATA / 'self_evolution' / SOURCE_ID
    episode = DATA / 'complete_sim/succeed' / SOURCE_ID
    trajectory = [json.loads(line) for line in (trial / 'trajectory.jsonl').read_text().splitlines()]
    if len(trajectory) != len(player.frames) or any(
        row['time'] != frame.timestep or row['phase'] != frame.phase or row['action'] != list(frame.nominal_action)
        for row, frame in zip(trajectory, player.frames)
    ):
        raise ValueError('Frozen reference is not an exact projection of the source trajectory')

    metadata = json.loads((episode / 'proprio_stats/metadata.json').read_text())
    outcome = json.loads((episode / 'outcome.json').read_text())
    episode_validation = validate(episode, trial, expected_success=True)
    replay_path = prior / 'stage1_replay.json'
    replay = json.loads(replay_path.read_text())
    recording_result = ROOT / replay['recording_report'] / 'result.json'
    recording = json.loads(recording_result.read_text())
    if not (
        replay['decision'] == 'PASS'
        and replay['task_success']
        and replay['recording_verified']
        and replay['reference_hash'] == nominal_manifest['reference_hash']
        and replay['nominal_action_hash'] == replay['executed_action_hash']
        and replay['planner_calls'] == replay['privileged_policy_reads'] == 0
        and replay['max_state_error_vs_generator'] == 0.0
        and recording['result'] == 'success'
        and recording['task_success']
        and outcome['success']
        and metadata['success']
    ):
        raise ValueError('Existing frozen replay or Episode validation is not valid')

    rows = export_rows(player.frames)
    native_sequence = runs([row['native_phase'] for row in rows])
    semantic_sequence = task_sequence([row['native_phase'] for row in rows])
    output = experiment / 'full_task_reference.jsonl'
    output.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))

    manifest = {
        'schema_version': 1,
        'stage': 'B',
        'validity': 'VALID',
        'hypothesis': 'SUPPORTED',
        'claims_allowed': ['The complete frozen nominal is executable without replanning and passes the full Episode validator.'],
        'claims_not_allowed': ['The reference establishes robustness to post-grasp perturbations or real-world transfer.'],
        'program_decision': 'CONTINUE',
        'source_episode': {
            'id': SOURCE_ID,
            'dataset_relative_path': f'complete_sim/succeed/{SOURCE_ID}',
            'trajectory_relative_path': f'self_evolution/{SOURCE_ID}/trajectory.jsonl',
            'trajectory_sha256': digest(trial / 'trajectory.jsonl'),
            'initial_state_sha256': digest(trial / 'initial_state.npz'),
            'metadata_sha256': digest(episode / 'proprio_stats/metadata.json'),
            'outcome_sha256': digest(episode / 'outcome.json'),
            'proprio_h5_sha256': digest(episode / 'proprio_stats/proprio_stats.h5'),
            'model_sha256': digest(episode / 'proprio_stats/model.xml'),
        },
        'reference': {
            'dataset_relative_path': str(output.relative_to(ROOT)),
            'sha256': digest(output),
            'source_frozen_reference_relative_path': nominal_manifest['dataset_relative_path'],
            'source_frozen_reference_sha256': nominal_manifest['sha256'],
            'reference_hash': nominal_manifest['reference_hash'],
            'nominal_action_hash': player.nominal_action_hash,
            'frames': len(rows),
            'duration_s': rows[-1]['timestep_s'],
            'timestep_s': 1 / 30,
            'action_dim': 14,
            'action_names': json.loads(reference_path.read_text())['action_names'],
            'action_units': ['rad'] * 6 + ['m'] + ['rad'] * 6 + ['m'],
            'action_semantics': 'Native dual-arm joint-position targets plus full gripper width; target active during interval ending at the sample.',
            'target_layout': {
                'left_joint_target_rad': [0, 6],
                'left_gripper_target_m': 6,
                'right_joint_target_rad': [7, 13],
                'right_gripper_target_m': 13,
            },
            'generation': 'Exact JSONL projection of the already frozen reference; no IK, planner, retiming, smoothing or trajectory regeneration.',
        },
        'phase_audit': {
            'native_sequence': native_sequence,
            'native_counts': dict(Counter(row['native_phase'] for row in rows)),
            'task_sequence': semantic_sequence,
            'required_sequence': list(REQUIRED_TASK_SEQUENCE),
        },
        'success_provenance': {
            'definition': metadata['success_definition'],
            'source_outcome_success': outcome['success'],
            'source_episode_validation': episode_validation,
            'frozen_replay_artifact': str(replay_path.relative_to(ROOT)),
            'frozen_replay_sha256': digest(replay_path),
            'frozen_replay_decision': replay['decision'],
            'recording_validator_artifact': str(recording_result.relative_to(ROOT)),
            'recording_validator_sha256': digest(recording_result),
            'recording_validator_result': recording['result'],
            'planner_calls': replay['planner_calls'],
            'privileged_policy_reads': replay['privileged_policy_reads'],
            'max_state_error_vs_generator': replay['max_state_error_vs_generator'],
        },
        'prohibited_operations': {
            'new_planner_calls': 0,
            'trajectory_regeneration': 0,
            'privileged_policy_reads': 0,
            'hardware_executions': 0,
            'real_data_reads': 0,
            'twin_mutations': 0,
        },
    }
    save(experiment / 'full_task_reference_manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

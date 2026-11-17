"""Archive an already stopped, explicitly reviewed real trial via native tools.

No hardware control and no automatic success inference. Original controller
outcomes remain unchanged; the separate RGB review supplies the physical label.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_shared_reference import compare
from learn import DATA, SIM, digest, save


def archive(index, review_path, reference, startup, *, capture=None,
            session='20261116_real_repeatability', output=None):
    batch = DATA / 'self_evolution/repeatability_cycle12'
    number = f'{index:06d}'
    identity = f'{session}_{number}'
    if capture is None:
        capture = batch / 'episodes' / identity / 'raw/task'
    else:
        capture = Path(capture)
        batch = capture.parent
    reference = Path(reference)
    output = Path(output) if output else ROOT / 'experiments/repeatability_cycle12' / number
    output.mkdir(parents=True, exist_ok=False)
    review = json.loads(review_path.read_text())
    original = json.loads((capture / 'execution_report.json').read_text())
    commands = compare(reference, capture)
    if review['physical_task_success']:
        assert original['completed_sequence'] and commands['entire_reference_sent']
        assert review['plug_retained_in_target_after_release_and_fold'] is True
    assert (capture / 'STOP').exists(), 'Capture/task has not ended'
    observed = dict(original, physical_task_success=review['physical_task_success'],
                    physical_success_verification=review,
                    original_execution_report_sha256=digest(capture / 'execution_report.json'))
    save(capture / 'observed_execution_report.json', observed)
    save(output / 'physical_review.json', review)
    save(output / 'commands.json', commands)
    domain = 'successed' if review['physical_task_success'] else 'failed'
    episode = DATA / 'complete_real' / domain / identity
    if episode.exists():
        raise FileExistsError(episode)
    python = SIM / '.venv-mujoco-hardware/bin/python'
    with (batch / f'export_{number}.log').open('w') as log:
        subprocess.run([str(python), str(SIM / 'scripts/export_plug_calibration.py'),
            '--capture', str(capture), '--root', str(DATA), '--session', session,
            '--episode', number, '--commands', str(capture / 'commanded_probe.csv'),
            '--purpose', 'yellow_pick_insert_attempt', '--execution-report',
            str(capture / 'observed_execution_report.json'), '--arms', 'right'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    target = episode / 'raw/task'
    target.parent.mkdir()
    capture.rename(target)
    capture.symlink_to(target, target_is_directory=True)
    view = batch / f'verify_{number}'
    for name, source in [('proprio_stats', episode / 'proprio_stats'), ('observations', episode)]:
        parent = view / name / session
        parent.mkdir(parents=True)
        (parent / number).symlink_to(source, target_is_directory=True)
    with (batch / f'verify_{number}.log').open('w') as log:
        subprocess.run([str(python), str(SIM / 'scripts/verify_mixed_plug_dataset.py'),
            '--root', str(view)], stdout=log, stderr=subprocess.STDOUT, check=True)
    verification = json.loads((view / 'dataset_manifest.json').read_text())
    save(episode / 'verification.json', verification)
    save(output / 'recording.json', verification)
    save(episode / 'pair_manifest.json', dict(reference_sha256=commands['reference_id'],
        reference_relative_path=str(reference.relative_to(DATA)),
        physical_review='raw/task/observed_execution_report.json'))
    rows = list(csv.DictReader((capture / 'commanded_probe.csv').open()))
    result = dict(experiment_id=identity, timestamp=datetime.now(timezone.utc).isoformat(),
        batch_index=index, environment='real', task='Fixed-policy pick, insert, release and fold',
        model='Deterministic frozen-reference execution; explicit Agent RGB outcome review',
        program_version=dict(reference_sha256=commands['reference_id'],
                             archive_program_sha256=digest(__file__)),
        initial_state_stratum=startup,
        task_gripper_effort_nm=original.get('task_gripper_effort_nm'),
        result='success' if review['physical_task_success'] else 'failure',
        task_success=review['physical_task_success'], recording_verified=True,
        frames=verification['verified_frames'], root_cause='UNKNOWN', root_cause_status='unknown',
        physical_review=review, reference_commands=commands['command_samples'],
        entire_reference_sent=commands['entire_reference_sent'],
        reference_execution_seconds=commands['real_logged_span_s'],
        nominal_reference_seconds=commands['reference_prefix_span_s'],
        execution_time=float(rows[-1]['monotonic'])-float(rows[0]['monotonic']),
        execution_llm_calls=0, tokens=0, research_orchestration_tokens=None,
        human_interventions=1, human_intervention_definition='One manual plug reset before this attempt; final supported shutdown counted separately',
        native_policy_training_eligible=False, accepted_metric_calibration=False,
        evidence=[dict(dataset_relative_path=str(p.relative_to(DATA)),sha256=digest(p)) for p in
            [episode / 'videos/cam_external_rgb.mp4', episode / 'videos/cam_right_wrist_rgb.mp4',
             episode / 'proprio_stats/proprio_stats.h5', episode / 'outcome.json',
             episode / 'verification.json', target / 'commanded_probe.csv']])
    save(output / 'result.json', result)
    (output / 'REPORT.md').write_text(
        f"# {identity}\n\n"
        f"Result: **{result['result']}**. Full recording verified: {result['frames']} frames. "
        f"Reference span: {result['reference_execution_seconds']:.3f}s; total command span: {result['execution_time']:.3f}s. "
        f"Startup stratum: {result['initial_state_stratum']}.\n\n"
        "The physical label is the explicit RGB review, separate from command completion. "
        "Original controller report and all raw evidence remain local with this Episode. "
        "Native policy-training eligibility remains false. No calibration or generalized reliability claim.\n")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=int)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--startup', choices=['cold_start','enabled_folded_zero'], required=True)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--session', default='20261116_real_repeatability')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = archive(args.index, args.review, args.reference, args.startup,
                         capture=args.capture, session=args.session, output=args.output)
        print(json.dumps({k:result[k] for k in ['experiment_id','result','frames','execution_time']}))
    except Exception as exc:
        output = args.output or Path(__file__).parent / f'{args.index:06d}'
        output.mkdir(exist_ok=True)
        save(output / 'archive_error.json', dict(result='infrastructure_error', error=str(exc)))
        (output / 'ARCHIVE_ERROR.md').write_text(f'# Archive failure\n\n{type(exc).__name__}: {exc}\n')
        raise

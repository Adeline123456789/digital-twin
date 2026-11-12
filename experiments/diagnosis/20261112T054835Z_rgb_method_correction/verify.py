"""Read-only regression evidence for RGB timestamps and existing input inventory.

No image tracking, parameter fitting, physics rollout or hardware connection.
"""
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from learn import DATA, SIM, digest, save, version
from observe_real_failures import camera_monotonic_times


def reference(path):
    return {'dataset_relative_path':str(path.relative_to(DATA)), 'sha256':digest(path)}


def main():
    started = time.monotonic()
    design = json.loads((ROOT/'experiments/research_stage2/round6/design.json').read_text())
    evidence = []
    for name in [design['train_episode']]+design['heldout_episodes']:
        ep = DATA/'complete_real/failed'/name
        command = ep/'raw/task/commanded_probe.csv'
        h5 = ep/'proprio_stats/proprio_stats.h5'
        camera = ep/'raw/task/right_wrist/camera.json'
        rows = list(csv.DictReader(command.open()))
        ct = np.array([float(r['monotonic']) for r in rows])
        phase = np.array([r['phase'] for r in rows])
        samples = json.loads(camera.read_text())['samples']
        with h5py.File(h5) as f:
            old = f['timestamp_monotonic'][:]
            corrected = camera_monotonic_times(f, 'cam_right_wrist')
            source_frames = f['camera/cam_right_wrist/source_frame'][:]
            rgb_unix = np.array([samples[int(i)]['rgb_device_ms']/1000 for i in source_frames])
            np.testing.assert_allclose(f['camera/cam_right_wrist/timestamp'][:], rgb_unix, rtol=0, atol=1e-6)
            np.testing.assert_allclose(corrected, rgb_unix+old-f['timestamp'][:], rtol=0, atol=1e-6)
            left_range = np.ptp(f['state/left_arm/joint_positions'][:], axis=0)
        assert all(samples[int(i)]['timestamp_domain']=='timestamp_domain.global_time' for i in source_frames)
        before, after = [np.clip(np.searchsorted(ct, t, side='right')-1, 0, len(ct)-1) for t in (old, corrected)]
        changes = np.flatnonzero(phase[before] != phase[after])
        evidence.append(dict(source_episode=name, frames=len(old),
            wrist_minus_master_ms_range=[float(np.min(corrected-old)*1000),float(np.max(corrected-old)*1000)],
            changed_command_row_count=int(np.sum(before != after)), changed_phase_count=len(changes),
            changed_phases=[dict(frame=int(i), before=phase[before[i]], after=phase[after[i]]) for i in changes],
            left_joint_range_deg=left_range.tolist(),
            source_frame_clock_check='PASS', evidence=[reference(p) for p in (command,h5,camera)]))

    captures = []
    for name in ('wrist_multiaxis_20261109_01','wrist_probe_01','wrist_probe_02'):
        p = DATA/'calibration'/name
        command = p/'commanded_probe.csv'
        rows = list(csv.DictReader(command.open()))
        q = np.array([[float(r[f'feedback_j{i}']) for i in range(1,7)] for r in rows])
        frames = sorted((p/'right_wrist').glob('*.jpg'))
        captures.append(dict(dataset_relative_path=str(p.relative_to(DATA)), command_rows=len(rows),
            duration_s=float(rows[-1]['monotonic'])-float(rows[0]['monotonic']),
            phases=list(dict.fromkeys(r['phase'] for r in rows)), right_rgb_jpegs=len(frames),
            measured_joint_range_deg=np.ptp(q, axis=0).tolist(),
            evidence=[reference(command)]+[reference(f) for f in (frames[0],frames[-1])]))

    previous_sources = json.loads((ROOT/'experiments/research_stage2/20261112T052851Z_68d83a86/result.json').read_text())['program_version']['source_sha256']
    assert previous_sources == version()['source_sha256']
    result = dict(experiment_id=Path(__file__).parent.name, timestamp=datetime.now(timezone.utc).isoformat(),
        task='Correct existing RGB observation clock and audit already collected calibration inputs',
        model='deterministic Python; literature/source assessment by Codex', environment='offline_archive',
        program_version={name:digest(ROOT/name) for name in ('scripts/observe_real_failures.py','scripts/validate_rgb_motion.py')},
        original_exporter_sha256=digest(SIM/'scripts/export_plug_calibration.py'),
        original_simulation_sources_unchanged_since_round6=True,
        protocol_version=digest(ROOT/'protocol/robot_problem_solving.md'),
        memory_version={'ER006':json.loads((ROOT/'memory/exploration_rules/ER006.json').read_text())['version']},
        initial_state={'input_design':'experiments/research_stage2/round6/design.json'},
        actions=['Compare HDF5 exposure times against original RGB source-frame metadata', 'Inventory existing RGB and measured joint records'],
        expected_effects=['Use the selected camera exposure to query command phases and motor feedback'],
        observed_effects=evidence, existing_calibration_inputs=captures,
        failure_signature='right_wrist_frames_aligned_to_external_camera_clock',
        root_cause='Observation scripts used the external-camera master timestamps for right-wrist frames',
        root_cause_status='verified', root_cause_scope='software timestamp selection only; physical failure cause remains UNKNOWN',
        repair={'shared_helper':'camera_monotonic_times', 'physics_changed':False, 'dataset_modified':False},
        result='success', task_success=None, validation_status='SOFTWARE_AND_ARCHIVE_VERIFIED_ONLY',
        new_research_round=False, research_rounds_used=6, calibration_fit_run=False,
        software_checks_reference='verification.json (separately executed; this script only checks archives)',
        tokens={'planner':0,'research_orchestrator':None}, execution_time=time.monotonic()-started,
        llm_inference_seconds=None, simulation_seconds=0, robot_execution_seconds=0, human_interventions=None,
        lessons_learned=['Audit existing calibration captures before assuming task failures are the only available data',
            'A failed fixed-ROI measurement is not evidence that RGB hand-eye calibration is impossible'],
        next_improvement='Use existing multiaxis RGB/feedback inputs to adapt an established hand-eye solver; do not promote depth-derived candidates')
    save(Path(__file__).with_name('result.json'),result)
    print(json.dumps({'alignment':[{k:v for k,v in r.items() if k!='evidence'} for r in evidence], 'captures':[{k:v for k,v in r.items() if k!='evidence'} for r in captures]},indent=2))


if __name__=='__main__': main()

"""Read a stopped right-arm capture into phase/event evidence; never infer success."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from evidence_cache import EvidenceCache
from recording_clock import map_camera_clock


def clock_evidence(samples):
    device = np.array([s['rgb_device_ms'] / 1000 for s in samples])
    receipt = np.array([s['host_monotonic'] for s in samples])
    elapsed_error = (device - device[0]) - (receipt - receipt[0])
    outside = np.flatnonzero(np.abs(elapsed_error) > 1 / 30)
    report = dict(frames=len(samples),
        max_elapsed_error_s=float(np.max(np.abs(elapsed_error))),
        first_outside_one_frame=None if not len(outside) else int(outside[0]),
        recorded_frame_number_gaps=int(sum(max(0, b['rgb_frame'] - a['rgb_frame'] - 1)
            for a, b in zip(samples, samples[1:]))))
    try:
        if any(s['timestamp_domain'] != 'timestamp_domain.global_time' for s in samples):
            raise ValueError('Native exporter requires global_time')
        mapped, mapping = map_camera_clock(samples)
        report.update(status='ACCEPTED_BY_CURRENT_MAPPING', mapping=mapping)
    except ValueError as exc:
        mapped = None
        report.update(status='REJECTED_BY_CURRENT_MAPPING', error=str(exc))
    return mapped, report


def summarize_capture(capture, cache):
    capture = Path(capture).resolve()
    if not (capture / 'STOP').exists():
        raise ValueError('Only stopped captures may be indexed')
    deps = dict(commands=capture / 'commanded_probe.csv',
        execution=capture / 'execution_report.json',
        external=capture / 'external/camera.json',
        right_wrist=capture / 'right_wrist/camera.json',
        analyzer=Path(__file__), clock=Path(__file__).with_name('recording_clock.py'))
    if (capture / 'observed_execution_report.json').exists():
        deps['review'] = capture / 'observed_execution_report.json'

    def compute():
        start = time.monotonic()
        with deps['commands'].open() as stream:
            rows = list(csv.DictReader(stream))
        execution = json.loads(deps['execution'].read_text())
        review = json.loads(deps.get('review', deps['execution']).read_text())
        cameras = {c: json.loads(deps[c].read_text())['samples']
                   for c in ('external', 'right_wrist')}
        read_seconds = time.monotonic() - start
        times = np.array([float(r['monotonic']) for r in rows])
        if not len(rows) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Command times must be finite and strictly increasing')
        start = time.monotonic()
        clocks = {c: clock_evidence(s) for c, s in cameras.items()}
        boundaries = [0] + [i for i in range(1, len(rows))
                           if rows[i]['phase'] != rows[i - 1]['phase']] + [len(rows)]
        phases = []
        for lo, hi in zip(boundaries, boundaries[1:]):
            row = rows[hi - 1]
            views = {}
            for camera, (mapped, _) in clocks.items():
                if mapped is None or not mapped[0] <= times[hi - 1] <= mapped[-1]:
                    continue
                frame = int(np.argmin(np.abs(mapped - times[hi - 1])))
                views[camera] = dict(path=f'{camera}/{frame:04d}.jpg',
                    exposure_minus_command_s=float(mapped[frame] - times[hi - 1]))
            phases.append(dict(phase=row['phase'], command_rows=[lo, hi - 1],
                elapsed_s=[float(times[lo] - times[0]), float(times[hi - 1] - times[0])],
                end_target_width_mm=float(row['target_gripper_open']) * 95,
                end_encoder_width_mm=float(row['feedback_gripper_open']) * 95,
                end_views=views))
        event = None
        if execution.get('insertion_stopped_frame'):
            frame = int(Path(execution['insertion_stopped_frame']).stem)
            event = dict(kind='NATIVE_INSERTION_STOP', phase='insert',
                image=f'right_wrist/{frame:04d}.jpg',
                image_window=[max(0, frame - 15), min(len(cameras['right_wrist']) - 1, frame + 15)],
                reason=execution.get('insertion_stop_reason'),
                source='Original controller event; not an inferred earliest physical onset')
        return dict(reference_id=execution.get('reference_id'),
            physical_task_success=review.get('physical_task_success'),
            physical_label_source='observed_execution_report.json' if 'review' in deps else 'execution_report.json',
            completed_sequence=execution.get('completed_sequence'),
            phases=phases, recorded_stop_event=event,
            camera_clocks={c: r for c, (_, r) in clocks.items()},
            root_cause='UNKNOWN',
            next_observation='Compare plug-to-finger pose from grasp through lift before selecting another insertion candidate.',
            timing=dict(input_read_seconds=read_seconds, analysis_seconds=time.monotonic() - start),
            limitations=['Saved RGB/encoder evidence only; no force or metric object-pose measurement.',
                'Rejected camera mapping suppresses command-to-image pairing; original event image remains available.',
                'Physical labels are copied from explicit prior reviews, never inferred from completion.',
                'This index does not revalidate video/HDF5 data or grant training eligibility.'])

    return cache.get('real_capture_summary_v1', deps, {}, compute)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = EvidenceCache(args.cache)
    try:
        result = summarize_capture(args.capture, cache)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    finally:
        cache.close()

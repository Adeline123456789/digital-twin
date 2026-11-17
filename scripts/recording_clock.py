"""Map continuous camera exposure time to the robot's monotonic clock.

The first host epoch/monotonic pair anchors the conversion. Later wall-clock
corrections must not move already continuous camera exposures by hundreds of ms.
"""
import numpy as np


def map_camera_clock(samples):
    device = np.array([s['rgb_device_ms'] / 1000 for s in samples])
    mono = np.array([s['host_monotonic'] for s in samples])
    wall = np.array([s['host_unix'] for s in samples])
    if len(samples) < 2 or not np.isfinite([device, mono, wall]).all():
        raise ValueError('Need finite camera and host clock pairs')
    elapsed_error = (device - device[0]) - (mono - mono[0])
    # shortcut: short 30fps recordings with a stable camera clock; reject a
    # device-clock rebase rather than silently fitting or warping the trajectory.
    if np.any(np.diff(device) <= 0) or np.max(np.abs(elapsed_error)) > 1 / 30:
        raise ValueError('Camera clock is not continuous relative to host monotonic time')
    mapped = (device - device[0]) + mono[0] + (device[0] - wall[0])
    previous = device + (mono - wall)
    report = dict(method='first_epoch_anchor_continuous_camera_v1',
        max_camera_receipt_elapsed_error_s=float(np.max(np.abs(elapsed_error))),
        largest_wallclock_step_s=float(np.diff(wall - mono)[np.argmax(np.abs(np.diff(wall - mono)))]),
        max_change_from_per_frame_wall_mapping_s=float(np.max(np.abs(previous - mapped))),
        initial_exposure_to_receipt_s=float(wall[0] - device[0]),
        scope='Initial epoch alignment retained; elapsed consistency checked against host receipts. No independent exposure-clock calibration.')
    return mapped, report

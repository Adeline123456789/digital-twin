import math


def aligned_samples(rows, privileged, sim_times, fps):
    """Compare native sim time, object position, phase and frame before pairing."""
    if not rows or len(rows) != len(privileged) or len(rows) != len(sim_times):
        return False
    for index, (row, extra, recorded) in enumerate(zip(rows, privileged, sim_times)):
        t = row['time']
        if not math.isfinite(t) or abs(t-recorded) > 1e-7 or not math.isfinite(recorded):
            return False
        if index and not 0 < t-rows[index-1]['time'] <= 1.5/fps:
            return False
        if extra['frame'] != index or row['phase'] != extra['phase'] or row['plug'] != extra['objects']['yellow_plug']['position']:
            return False
        if not all(math.isfinite(x) for vector in (row['plug'], row['grip']) for x in vector):
            return False
    return True

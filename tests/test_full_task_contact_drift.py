import numpy as np

from robot_agent.full_task_residual_env import sample_contact_drift


def test_contact_drift_sampler_uses_only_frozen_intervals():
    distribution = {
        'phase': 'lift_middle',
        'force_n': 10.,
        'duration_ms': 50.,
        'angle_rad': {'intervals': [[-2., -1.], [.5, 1.]], 'weights': [.25, .75]},
    }
    rng = np.random.default_rng(61001)
    rows = [sample_contact_drift(distribution, rng, f'train_{i}') for i in range(100)]

    assert all(row['phase'] == 'lift_middle' for row in rows)
    assert all(row['force_n'] == 10. and row['duration_ms'] == 50. for row in rows)
    assert all(-2 <= row['angle_rad'] <= -1 or .5 <= row['angle_rad'] <= 1 for row in rows)
    assert len({row['angle_rad'] for row in rows}) == len(rows)

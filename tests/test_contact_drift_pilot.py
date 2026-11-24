import numpy as np

from scripts.pilot_contact_drift import CONDITION_COUNT, pilot_conditions, rotation_angle_deg


def test_contact_drift_pilot_domain_is_frozen_and_bounded():
    conditions = pilot_conditions()

    assert len(conditions) == CONDITION_COUNT
    assert len({row['condition_id'] for row in conditions}) == CONDITION_COUNT
    assert {row['phase'] for row in conditions} == {
        'lift_middle', 'transport_early', 'transport_late',
    }
    assert all(5 <= row['force_n'] <= 25 for row in conditions)
    assert all(50 <= row['duration_ms'] <= 200 for row in conditions)
    assert all(-np.pi <= row['angle_rad'] <= np.pi for row in conditions)


def test_rotation_angle_uses_relative_rotation():
    before = np.eye(3)
    angle = np.deg2rad(10)
    after = np.array([[np.cos(angle), -np.sin(angle), 0.],
                      [np.sin(angle), np.cos(angle), 0.],
                      [0., 0., 1.]])

    assert np.isclose(rotation_angle_deg(before, after), 10.)

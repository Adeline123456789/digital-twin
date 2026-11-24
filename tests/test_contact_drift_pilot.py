import numpy as np

from scripts.pilot_contact_drift import (
    CONDITION_COUNT,
    FREEZE_CONDITION_COUNT,
    FOCUSED_CONDITION_COUNT,
    freeze_validation_conditions,
    focused_conditions,
    pilot_conditions,
    rotation_angle_deg,
    tuning_conditions,
)


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


def test_focused_pilot_is_continuous_angle_lift_calibration():
    conditions = focused_conditions()

    assert len(conditions) == FOCUSED_CONDITION_COUNT
    assert {row['phase'] for row in conditions} == {'lift_middle'}
    assert {row['force_n'] for row in conditions} == {10.}
    assert {row['duration_ms'] for row in conditions} == {50.}
    assert len({row['angle_rad'] for row in conditions}) == FOCUSED_CONDITION_COUNT
    assert all(-np.pi <= row['angle_rad'] <= np.pi for row in conditions)


def test_freeze_validation_samples_selected_continuous_interval():
    conditions = freeze_validation_conditions()

    assert len(conditions) == FREEZE_CONDITION_COUNT
    assert {row['force_n'] for row in conditions} == {8.}
    assert all(-.28 <= row['angle_rad'] <= -.268 or 1.05 <= row['angle_rad'] <= 1.08
               for row in conditions)
    assert sum(-.28 <= row['angle_rad'] <= -.268 for row in conditions) == FREEZE_CONDITION_COUNT // 2
    assert sum(1.05 <= row['angle_rad'] <= 1.08 for row in conditions) == FREEZE_CONDITION_COUNT // 2
    assert len({row['angle_rad'] for row in conditions}) == FREEZE_CONDITION_COUNT


def test_tuning_wave_compares_two_lower_impulses():
    conditions = tuning_conditions()

    assert len(conditions) == 24
    assert {row['force_n'] for row in conditions} == {8., 9.}
    assert all(-.35 <= row['angle_rad'] <= -.2 or .9 <= row['angle_rad'] <= 1.1
               for row in conditions)

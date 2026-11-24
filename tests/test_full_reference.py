from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_full_nominal import export_rows, task_sequence


def test_full_reference_maps_dual_arm_targets_and_required_phase_order():
    phases = ['descend', 'grasp', 'lift', 'transport', 'align', 'insert', 'release', 'retreat', 'fold']
    frames = [SimpleNamespace(timestep=(index + 1) / 30, phase=phase, nominal_action=tuple(range(14)))
              for index, phase in enumerate(phases)]

    rows = export_rows(frames)

    assert rows[0]['left_joint_target_rad'] == list(range(6))
    assert rows[0]['left_gripper_target_m'] == 6
    assert rows[0]['right_joint_target_rad'] == list(range(7, 13))
    assert rows[0]['right_gripper_target_m'] == 13
    assert task_sequence(phases)[:8] == [
        'APPROACH', 'GRASP', 'LIFT', 'TRANSPORT', 'ALIGN', 'INSERT', 'RELEASE', 'RETREAT'
    ]

    with pytest.raises(ValueError, match='phase order'):
        task_sequence(['descend', 'grasp', 'insert'])

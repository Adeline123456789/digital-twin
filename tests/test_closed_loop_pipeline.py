"""Unit and MuJoCo integration tests for the unified 4-stage closed-loop plug-insert pipeline."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from robot_agent.bimanual_closed_loop import OnlineSupervisor  # noqa: E402
from robot_agent.compliant_residual import (  # noqa: E402
    HybridCompliantInsertController,
    SmoothResidual4D,
    inject_right_arm_residual,
)
from robot_agent.scene_retarget import detect_strip_features_rgb, pixel_to_world_at_z  # noqa: E402
from robot_agent.wrist_inhand import WristInHandCompensator  # noqa: E402
from execute_real_closed_loop import dry_run_preview  # noqa: E402


class ClosedLoopPipelineTests(unittest.TestCase):
    def test_smooth_residual_4d_bounds_and_left_invariance(self):
        filt = SmoothResidual4D()
        prev = np.zeros(4)
        for step in range(40):
            req = np.array([0.02, -0.02, -0.01 if step < 10 else 0.02, 0.5])
            pos = filt.update_metric(req)
            self.assertTrue(np.all(np.abs(pos[:2]) <= 0.0042 + 1e-9))
            self.assertGreaterEqual(pos[2], 0.0)
            self.assertLessEqual(pos[2], 0.0040 + 1e-9)
            vel = np.abs(pos - prev) * 30.0
            self.assertTrue(np.all(vel[:2] <= 0.0240 + 1e-4))
            prev = pos

        nominal = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1, 0.06] + [0.0] * 6 + [0.027])
        lower = np.array([-3.0] * 6 + [0.0] + [-3.0] * 6 + [0.0])
        upper = np.array([3.0] * 6 + [0.08] + [3.0] * 6 + [0.08])
        out, dq = inject_right_arm_residual(nominal, pos, np.eye(6), lower, upper)
        np.testing.assert_allclose(out[:7], nominal[:7], atol=0.0)
        self.assertEqual(out[13], nominal[13])
        self.assertTrue(np.all(np.abs(dq) <= 0.045 + 1e-9))

    def test_jam_detector_unloads_and_advances_probe(self):
        ctrl = HybridCompliantInsertController(jam_ez_threshold_m=-0.0058, unload_height_m=0.0020, vertical_clear_steps=2, air_shift_steps=3)
        ctrl.reset(initial_xy_bias_m=(0.0, 0.0), lift_gripper_width_m=0.038)
        # Simulate downward rim jam at progress=0.75: e_z = -0.008 m < -0.0058 m
        q_meas = np.zeros(6)
        q_cmd = np.array([0.0, 0.0, -0.008, 0.0, 0.0, 0.0])
        J = np.eye(6)
        res, pause, telem = ctrl.step(
            phase="insert",
            insert_progress=0.75,
            q_right_6=q_meas,
            gripper_width_m=0.038,
            commanded_right_6=q_cmd,
            jacobian_6x6=J,
        )
        self.assertTrue(telem.jam_detected)
        self.assertTrue(pause)
        self.assertEqual(ctrl.probe_idx, 1)
        self.assertGreater(res[2], 0.0)

    def test_wrist_inhand_empty_grasp_rejected(self):
        comp = WristInHandCompensator()
        blank = np.full((480, 640, 3), 200, dtype=np.uint8)
        check = comp.evaluate_post_lift(blank, gripper_width_m=0.0270, gripper_rotation_world=np.eye(3))
        self.assertFalse(check.retained)
        self.assertEqual(check.status, "EMPTY_OR_DROPPED")

    def test_online_supervisor_persists_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bias.json"
            sup = OnlineSupervisor(bias_path=path, step_gain=0.8)
            sup.update_from_episode(seated_success=True, winning_residual_xy_m=(0.0015, -0.0010), jam_events=2)
            reloaded = OnlineSupervisor(bias_path=path)
            self.assertAlmostEqual(reloaded.current_bias()[0], 0.0012, places=6)
            self.assertAlmostEqual(reloaded.current_bias()[1], -0.0008, places=6)

    def test_socket_panels_survive_a_blue_cast_on_the_white_strip(self):
        hsv = np.zeros((480, 640, 3), dtype=np.uint8)
        hsv[238:285, 255:370] = (104, 54, 175)
        for x in (273, 294, 315, 336):
            hsv[259:276, x-9:x+10] = (104, 145, 161)
            hsv[248:255, x-4:x+5] = (104, 132, 160)
        features = detect_strip_features_rgb(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        self.assertEqual(len(features["panels_px"]), 4)
        np.testing.assert_allclose(features["outlet3_px"], [336., 267.], atol=.1)

    def test_scene_retarget_ray_plane_and_dry_run_preview(self):
        K = np.array([[387.0, 0.0, 324.0], [0.0, 386.5, 238.0], [0.0, 0.0, 1.0]])
        T_wc = np.eye(4)
        T_wc[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T_wc[:3, 3] = np.array([0.30, 0.0, 0.75])
        w = pixel_to_world_at_z((324.0, 238.0), 0.0283, K, T_wc)
        np.testing.assert_allclose(w, [0.30, 0.0, 0.0283], atol=1e-6)

        sim_root = Path("/media/midea/F496AC8096AC4548/piper_simulation")
        preview = dry_run_preview(sim_root / "configs/tasks/real_connector_calibrated.yaml", (4.0, -3.0), (-2.0, 2.5))
        self.assertEqual(preview["hardware_commands"], 0)
        self.assertTrue(preview["within_joint_limits"])
        self.assertTrue(preview["ready_for_hardware"])
        self.assertAlmostEqual(preview["initial_gripper_width_mm"], 40.0, places=1)


if __name__ == "__main__":
    unittest.main()

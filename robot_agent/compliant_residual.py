"""Stage 2: Hybrid compliant residual insertion controller (zero force sensor).

Combines:
1. SmoothResidual4D critically-damped analytical filter over (dx, dy, dz_unload, dpitch),
   guaranteeing position, velocity, acceleration, and joint-delta safety bounds.
2. Pure-NumPy deployable Residual MLP loader (`NumpyResidualMLP`) so both PyTorch training
   environments and lightweight hardware control environments run identical inference.
3. Zero-drag quasi-static rim-contact gate & concentric micro-probe search (no 6-axis F/T sensor):
   - Tests entry at quasi-static dwell points (earth prong gate @ 17.3 mm, blade gate @ 15.3 mm)
     where servo velocity lag vanishes.
   - On rim block, lifts vertically to free air (21.5 mm) BEFORE shifting XY to the next probe,
     eliminating lateral friction drag and plug tilt ratcheting inside the parallel jaws.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np


class NumpyResidualMLP:
    """Pure-NumPy inference wrapper for DeployableResidualMLP (.npz weights, zero torch dependency)."""

    def __init__(self, w0: np.ndarray, b0: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
        self.w0 = np.asarray(w0, dtype=np.float32)
        self.b0 = np.asarray(b0, dtype=np.float32)
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        arr = np.asarray(obs, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        h0 = np.tanh(arr @ self.w0.T + self.b0)
        h1 = np.tanh(h0 @ self.w1.T + self.b1)
        out = np.tanh(h1 @ self.w2.T + self.b2)
        return (out[0] if single else out), None


def load_deployed_residual_policy(npz_path: Path) -> NumpyResidualMLP | None:
    """Load pure-NumPy residual MLP weights if available."""
    if not npz_path.is_file():
        return None
    with np.load(npz_path, allow_pickle=False) as data:
        return NumpyResidualMLP(data["w0"], data["b0"], data["w1"], data["b1"], data["w2"], data["b2"])


class SmoothResidual4D:
    """Exact critically-damped second-order filter on (dx_m, dy_m, dz_m, dpitch_rad)."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float] = (0.0042, 0.0042, 0.0040, np.deg2rad(2.0)),
        max_speeds: tuple[float, float, float, float] = (0.0240, 0.0240, 0.0120, np.deg2rad(8.0)),
        max_accels: tuple[float, float, float, float] = (0.950, 0.950, 0.250, np.deg2rad(80.0)),
        dt: float = 1.0 / 30.0,
    ):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.max_speeds = np.asarray(max_speeds, dtype=np.float64)
        self.max_accels = np.asarray(max_accels, dtype=np.float64)
        self.dt = float(dt)
        self.omega = np.minimum(
            self.max_speeds * math.e / (2.0 * self.bounds),
            np.sqrt(self.max_accels / (4.0 * self.bounds)),
        )
        self.position = np.zeros(4, dtype=np.float64)
        self.velocity = np.zeros(4, dtype=np.float64)

    def reset(self) -> None:
        self.position[:] = 0.0
        self.velocity[:] = 0.0

    def update_metric(self, desired_metric: np.ndarray) -> np.ndarray:
        """Step filter toward desired [dx_m, dy_m, dz_m, dpitch_rad] within bounds."""
        arr = np.asarray(desired_metric, dtype=np.float64).reshape(4)
        if not np.isfinite(arr).all():
            raise ValueError("Residual target must be finite")
        target = np.clip(arr, -self.bounds, self.bounds)
        # dz_unload is strictly non-negative (never pushes deeper than nominal socket bottom!)
        target[2] = max(0.0, target[2])
        error = self.position - target
        transient = self.velocity + self.omega * error
        decay = np.exp(-self.omega * self.dt)
        self.position = target + (error + transient * self.dt) * decay
        self.velocity = (self.velocity - self.omega * transient * self.dt) * decay
        return self.position.copy()


def damped_cartesian_to_joints(
    jacobian_6x6: np.ndarray,
    delta_xy_z_pitch: np.ndarray,
    damping: float = 1e-4,
    max_joint_delta_rad: float = 0.045,
) -> np.ndarray:
    """Map [dx, dy, dz, dpitch_world_y] into a bounded 6-joint angle correction."""
    J = np.asarray(jacobian_6x6, dtype=np.float64).reshape(6, 6)
    twist = np.zeros(6, dtype=np.float64)
    twist[:3] = delta_xy_z_pitch[:3]
    twist[4] = delta_xy_z_pitch[3]  # pitch about world Y
    dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), twist)
    return np.clip(dq, -max_joint_delta_rad, max_joint_delta_rad)


def inject_right_arm_residual(
    nominal_action_14: np.ndarray,
    delta_xy_z_pitch: np.ndarray,
    jacobian_6x6: np.ndarray,
    lower_14: np.ndarray,
    upper_14: np.ndarray,
    max_joint_delta_rad: float = 0.045,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject bounded right-arm residual into 14-DoF dual-arm command; left arm stays untouched."""
    action = np.asarray(nominal_action_14, dtype=np.float64).copy()
    dq = damped_cartesian_to_joints(jacobian_6x6, delta_xy_z_pitch, max_joint_delta_rad=max_joint_delta_rad)
    action[7:13] = np.clip(action[7:13] + dq, lower_14[7:13], upper_14[7:13])
    return action, dq


def build_probe_candidates(max_radius_m: float = 0.0038) -> list[np.ndarray]:
    """Sequential small-chord circle-covering XY probes accounting for ~0.85x PD compliance transmission.

    - Probe 0 at r=0 covers |e| <= 0.80 mm.
    - Ring 1 (9 pts at r_cmd=1.72 mm -> r_plug~1.46 mm) covers 0.66..2.26 mm in <= 9 sequential 1.1 mm chords!
    - Ring 2 (12 pts at r_cmd=2.75 mm -> r_plug~2.34 mm) extends coverage to > 3.1 mm.
    """
    candidates = [np.zeros(2, dtype=np.float64)]
    rings = [
        (0.00172, 9, 0.0),
        (0.00275, 12, np.pi / 12.0),
    ]
    for r, n, phase0 in rings:
        if r > max_radius_m + 1e-9:
            continue
        for k in range(n):
            theta = phase0 + 2.0 * np.pi * k / n
            candidates.append(np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float64))
    return candidates


@dataclass
class InsertStepTelemetry:
    jam_detected: bool
    unloading: bool
    probe_index: int
    cartesian_error_mm: tuple[float, float, float]
    applied_residual_mm_deg: tuple[float, float, float, float]
    seated_ready: bool


class HybridCompliantInsertController:
    """Online compliant controller for align/insert without 6-axis force sensing."""

    def __init__(
        self,
        *,
        learned_policy=None,
        jam_ez_threshold_m: float = -0.0021,
        seated_ez_max_abs_m: float = 0.0020,
        slip_aperture_delta_m: float = 0.0025,
        contact_deflection_threshold_m: float = 0.00048,
        max_rim_cmd_h_m: float = 0.0215,
        unload_height_m: float = 0.0,
        vertical_clear_steps: int = 11,
        air_shift_steps: int = 10,
        max_probes: int = 28,
        dt: float = 1.0 / 30.0,
    ):
        self.learned_policy = learned_policy
        self.jam_ez_threshold_m = float(jam_ez_threshold_m)
        self.seated_ez_max_abs_m = float(seated_ez_max_abs_m)
        self.slip_aperture_delta_m = float(slip_aperture_delta_m)
        self.contact_deflection_threshold_m = float(contact_deflection_threshold_m)
        self.max_rim_cmd_h_m = float(max_rim_cmd_h_m)
        self.unload_height_m = float(unload_height_m)
        self.vertical_clear_steps = int(vertical_clear_steps)
        self.air_shift_steps = int(air_shift_steps)
        self.max_probes = int(max_probes)
        self.smoother = SmoothResidual4D(dt=dt)
        self.base_probes = build_probe_candidates()
        self.reset()

    def reset(self, initial_xy_bias_m: tuple[float, float] = (0.0, 0.0), lift_gripper_width_m: float = 0.038) -> None:
        self.smoother.reset()
        self.initial_xy = np.asarray(initial_xy_bias_m, dtype=np.float64).copy()
        self.active_xy = self.initial_xy.copy()
        self.pending_xy = self.initial_xy.copy()
        self.lift_gripper_width_m = float(lift_gripper_width_m)
        self.probe_idx = 0
        self.clear_steps_left = 0
        self.shift_steps_left = 0
        self.steps_since_unload = 999
        self.jam_events = 0
        self.seated_counter = 0
        self.ordered_probes = [self.initial_xy + p for p in self.base_probes]
        self.learned_hint_applied = False

    def apply_learned_hint(self, policy_obs: np.ndarray) -> np.ndarray | None:
        """Query optional Residual MLP policy at pre-insert ONLY when post-lift wrist features show in-hand shift."""
        if self.learned_policy is None or self.learned_hint_applied:
            return None
        self.learned_hint_applied = True
        # Last 4 dims of 499-D student obs are [wrist_dx, wrist_dy, vis, conf]; in free air without
        # wrist shift, extrinsic socket bias is unobservable until first rim contact, so keep isotropic rings!
        if len(policy_obs) >= 4 and float(np.linalg.norm(policy_obs[-4:-2])) < 0.060:
            return None
        raw_action, _ = self.learned_policy.predict(policy_obs, deterministic=True)
        hint_xy = np.clip(np.asarray(raw_action[:2], dtype=np.float64), -1.0, 1.0) * 0.0020
        center = np.clip(self.initial_xy + hint_xy, -0.0040, 0.0040)
        # Keep nominal (initial_xy) as Probe 0 so clean/aligned episodes never spoil;
        # on first feather-touch rim contact, Probe 1 jumps directly to the learned MLP prediction!
        ranked = sorted(self.base_probes[1:], key=lambda p: float(np.linalg.norm((self.initial_xy + p) - center)))
        self.ordered_probes = [self.initial_xy, center, *[np.clip(self.initial_xy + p, -0.0040, 0.0040) for p in ranked]]
        self.active_xy = self.ordered_probes[0].copy()
        self.pending_xy = self.active_xy.copy()
        return hint_xy

    def trigger_rim_backoff(self) -> bool:
        """Advance to next concentric XY probe using two-stage zero-drag vertical lift -> air XY shift."""
        if self.probe_idx >= min(self.max_probes, len(self.ordered_probes) - 1):
            return False
        self.jam_events += 1
        self.probe_idx += 1
        self.pending_xy = np.clip(self.ordered_probes[self.probe_idx], -0.0040, 0.0040)
        self.clear_steps_left = self.vertical_clear_steps
        self.shift_steps_left = self.air_shift_steps
        self.steps_since_unload = 0
        self.seated_counter = 0
        return True

    def step(
        self,
        *,
        phase: str,
        insert_progress: float,
        q_right_6: np.ndarray,
        gripper_width_m: float,
        commanded_right_6: np.ndarray,
        jacobian_6x6: np.ndarray,
        policy_obs: np.ndarray | None = None,
        actual_height_above_seated_m: float | None = None,
        cmd_height_above_seated_m: float | None = None,
        free_air_lag_m: float | None = None,
    ) -> tuple[np.ndarray, bool, InsertStepTelemetry]:
        """Compute smoothed 4D residual [dx, dy, dz_unload, dpitch] and whether nominal insert clock should pause."""
        eq = np.asarray(commanded_right_6, dtype=np.float64) - np.asarray(q_right_6, dtype=np.float64)
        J = np.asarray(jacobian_6x6, dtype=np.float64).reshape(6, 6)
        e_xyz = J[:3, :] @ eq
        ez = float(e_xyz[2])
        grip_delta = abs(float(gripper_width_m) - self.lift_gripper_width_m)

        pause_nominal_descent = False
        jam_now = False

        if phase == "align":
            if policy_obs is not None:
                self.apply_learned_hint(policy_obs)
            desired = np.array([self.active_xy[0], self.active_xy[1], 0.0, 0.0], dtype=np.float64)
        elif phase == "insert":
            if self.clear_steps_left > 0:
                # Stage A of backoff: keep previous XY fixed while nominal cursor lifts prong vertically off rim!
                self.clear_steps_left -= 1
                if self.clear_steps_left == 0:
                    self.active_xy = self.pending_xy.copy()
                pause_nominal_descent = True
                desired = np.array([self.active_xy[0], self.active_xy[1], self.unload_height_m, 0.0], dtype=np.float64)
            elif self.shift_steps_left > 0:
                # Stage B of backoff: prong is 2+ mm in free air above rim; glide XY smoothly to next probe!
                self.shift_steps_left -= 1
                pause_nominal_descent = True
                desired = np.array([self.active_xy[0], self.active_xy[1], self.unload_height_m, 0.0], dtype=np.float64)
            else:
                self.steps_since_unload += 1
                if actual_height_above_seated_m is not None and cmd_height_above_seated_m is not None:
                    height_lag_m = actual_height_above_seated_m - cmd_height_above_seated_m
                    baseline_lag = 0.00020 if free_air_lag_m is None else float(free_air_lag_m)
                    contact_deflection_m = height_lag_m - baseline_lag
                    # Self-calibrating feather-touch rim detector:
                    # While prongs are above full slot engagement (actual_h >= 15.0 mm) and descending below 21.5 mm,
                    # any +0.48 mm excess deflection over measured free-air lag triggers immediate vertical lift!
                    rim_blocked = (
                        cmd_height_above_seated_m <= self.max_rim_cmd_h_m
                        and actual_height_above_seated_m >= 0.0150
                        and contact_deflection_m >= self.contact_deflection_threshold_m
                        and self.steps_since_unload >= 6
                    )
                else:
                    rim_blocked = (
                        insert_progress >= 0.45
                        and self.steps_since_unload >= 10
                        and (ez < self.jam_ez_threshold_m or grip_delta > self.slip_aperture_delta_m)
                    )

                if rim_blocked and self.trigger_rim_backoff():
                    jam_now = True
                    pause_nominal_descent = True
                    desired = np.array([self.active_xy[0], self.active_xy[1], self.unload_height_m, 0.0], dtype=np.float64)
                else:
                    desired = np.array([self.active_xy[0], self.active_xy[1], 0.0, 0.0], dtype=np.float64)
                    at_bottom = (
                        (actual_height_above_seated_m <= 0.0025)
                        if actual_height_above_seated_m is not None
                        else (insert_progress >= 0.96 and ez > -self.seated_ez_max_abs_m)
                    )
                    if insert_progress >= 0.96 and at_bottom:
                        self.seated_counter += 1
                    else:
                        self.seated_counter = max(0, self.seated_counter - 1)
        elif phase == "release":
            # Hold winning XY offset steady while fingers open so pins experience zero side-load!
            desired = np.array([self.active_xy[0], self.active_xy[1], 0.0, 0.0], dtype=np.float64)
        else:
            desired = np.zeros(4, dtype=np.float64)

        smoothed = self.smoother.update_metric(desired)
        unloading_active = (self.clear_steps_left > 0) or (self.shift_steps_left > 0)
        telemetry = InsertStepTelemetry(
            jam_detected=jam_now,
            unloading=unloading_active,
            probe_index=self.probe_idx,
            cartesian_error_mm=(float(e_xyz[0] * 1000.0), float(e_xyz[1] * 1000.0), float(e_xyz[2] * 1000.0)),
            applied_residual_mm_deg=(
                float(smoothed[0] * 1000.0),
                float(smoothed[1] * 1000.0),
                float(smoothed[2] * 1000.0),
                float(np.rad2deg(smoothed[3])),
            ),
            seated_ready=self.seated_counter >= 6,
        )
        return smoothed, pause_nominal_descent, telemetry

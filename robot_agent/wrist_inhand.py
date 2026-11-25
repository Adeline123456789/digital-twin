"""Stage 1: Post-lift wrist RGB in-hand verification and pose compensation.

Operates strictly on deployable right-wrist RGB and measured gripper width (m).
Never accesses simulator object poses or contact structures.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .full_task_observation import WristRelativeFeatures, wrist_relative_features


@dataclass(frozen=True)
class WristInHandCheck:
    status: str  # 'RETENTION_OK' | 'EMPTY_OR_DROPPED' | 'UNRESOLVED'
    retained: bool
    gripper_width_m: float
    yellow_area_px: float
    features: WristRelativeFeatures
    world_xy_correction_m: tuple[float, float]
    yaw_correction_deg: float
    reason: str


def _yellow_contour_stats(rgb: np.ndarray) -> tuple[float, float]:
    """Return (largest_yellow_area_px, contour_orientation_deg) in wrist RGB."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (15, 75, 35), (42, 255, 255))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0
    best = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    if area < 64.0:
        return area, 0.0
    (_, _), (rw, rh), ang = cv2.minAreaRect(best)
    if rw < rh:
        ang += 90.0
    if ang > 45.0:
        ang -= 90.0
    elif ang < -45.0:
        ang += 90.0
    return area, float(ang)


class WristInHandCompensator:
    """Calibrates horizontal pixel-to-gripper-frame sensitivity from camera geometry."""

    def __init__(
        self,
        *,
        reference_rel_xy: tuple[float, float] = (-0.04474, 0.00346),
        reference_angle_deg: float = 0.0,
        min_retained_width_m: float = 0.0185,
        max_retained_width_m: float = 0.0520,
        min_yellow_area_px: float = 350.0,
        max_xy_correction_m: float = 0.0045,
        max_yaw_correction_deg: float = 4.0,
    ):
        self.reference_rel_xy = np.asarray(reference_rel_xy, dtype=np.float64)
        self.reference_angle_deg = float(reference_angle_deg)
        self.min_retained_width_m = float(min_retained_width_m)
        self.max_retained_width_m = float(max_retained_width_m)
        self.min_yellow_area_px = float(min_yellow_area_px)
        self.max_xy_correction_m = float(max_xy_correction_m)
        self.max_yaw_correction_deg = float(max_yaw_correction_deg)

        # Calibrated from horizontal-grasp pinhole geometry at z_plug_offset=20mm:
        # +1mm in gripper_y (finger-closing axis) -> d(rel_x) = -0.00511
        # +1mm in gripper_z (lateral finger-width axis) -> d(rel_y) = -0.00455
        self.scale_grip_y_per_rel_x = -0.001 / 0.00511
        self.scale_grip_z_per_rel_y = -0.001 / 0.00455

    def calibrate_reference(self, nominal_lift_rgb: np.ndarray) -> WristRelativeFeatures:
        """Lock onto the nominal post-lift wrist RGB appearance."""
        wf = wrist_relative_features(nominal_lift_rgb)
        _, ang = _yellow_contour_stats(nominal_lift_rgb)
        if wf.confidence > 0.2:
            self.reference_rel_xy = np.asarray(wf.relative_xy, dtype=np.float64)
            self.reference_angle_deg = ang
        return wf

    def evaluate_post_lift(
        self,
        wrist_rgb: np.ndarray,
        gripper_width_m: float,
        gripper_rotation_world: np.ndarray,
        *,
        apply_metric_offset: bool = True,
    ) -> WristInHandCheck:
        """Verify grasp retention after lift and compute world-XY target compensation."""
        width = float(gripper_width_m)
        area, angle_deg = _yellow_contour_stats(wrist_rgb)
        wf = wrist_relative_features(wrist_rgb)

        if width < self.min_retained_width_m:
            return WristInHandCheck(
                status="EMPTY_OR_DROPPED",
                retained=False,
                gripper_width_m=width,
                yellow_area_px=area,
                features=wf,
                world_xy_correction_m=(0.0, 0.0),
                yaw_correction_deg=0.0,
                reason=f"Gripper closed empty ({width * 1000:.2f} mm < {self.min_retained_width_m * 1000:.2f} mm)",
            )
        if width > self.max_retained_width_m or area < self.min_yellow_area_px:
            return WristInHandCheck(
                status="EMPTY_OR_DROPPED",
                retained=False,
                gripper_width_m=width,
                yellow_area_px=area,
                features=wf,
                world_xy_correction_m=(0.0, 0.0),
                yaw_correction_deg=0.0,
                reason=f"Plug missing or dropped in wrist view (width={width * 1000:.2f} mm, yellow_area={area:.0f} px)",
            )
        if not apply_metric_offset:
            return WristInHandCheck(
                status="RETENTION_OK",
                retained=True,
                gripper_width_m=width,
                yellow_area_px=area,
                features=wf,
                world_xy_correction_m=(0.0, 0.0),
                yaw_correction_deg=0.0,
                reason="Plug retention verified via wrist RGB + aperture (metric wrist offset uncalibrated on real camera)",
            )
        if wf.confidence <= 0.15 or wf.yellow_center is None:
            return WristInHandCheck(
                status="UNRESOLVED",
                retained=True,
                gripper_width_m=width,
                yellow_area_px=area,
                features=wf,
                world_xy_correction_m=(0.0, 0.0),
                yaw_correction_deg=0.0,
                reason="Plug retained by aperture/yellow area, but finger pair partially occluded; zero wrist offset applied",
            )

        d_rel = np.asarray(wf.relative_xy, dtype=np.float64) - self.reference_rel_xy
        # Estimate plug displacement relative to gripper in gripper local horizontal axes (y, z)
        d_grip_y = float(d_rel[0] * self.scale_grip_y_per_rel_x)
        d_grip_z = float(d_rel[1] * self.scale_grip_z_per_rel_y)
        d_grip_local = np.array([0.0, d_grip_y, d_grip_z], dtype=np.float64)

        R_wg = np.asarray(gripper_rotation_world, dtype=np.float64).reshape(3, 3)
        plug_shift_world = R_wg @ d_grip_local
        # If plug shifted by +delta in world relative to gripper, gripper must move by -delta over socket!
        corr_xy = np.clip(-plug_shift_world[:2], -self.max_xy_correction_m, self.max_xy_correction_m)
        corr_yaw = float(np.clip(-(angle_deg - self.reference_angle_deg) * 0.35, -self.max_yaw_correction_deg, self.max_yaw_correction_deg))

        return WristInHandCheck(
            status="RETENTION_OK",
            retained=True,
            gripper_width_m=width,
            yellow_area_px=area,
            features=wf,
            world_xy_correction_m=(float(corr_xy[0]), float(corr_xy[1])),
            yaw_correction_deg=corr_yaw,
            reason="Plug retained; wrist RGB relative offset measured",
        )

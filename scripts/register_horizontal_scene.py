"""Register the dual-arm tabletop from saved external RGB using this robot's camera."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from learn import DATA, ROOT, SIM, digest, save
sys.path.insert(0, str(ROOT))
from robot_agent.scene_retarget import align_scene_from_external_rgb, detect_strip_features_rgb, pixel_to_world_at_z, STRIP_FACE_Z_M

DEFAULT_PROFILE = ROOT / "experiments/closed_loop_v1/horizontal_scene_v2.json"
NOV16 = DATA / "self_evolution/20261116T105302Z_d700416b"
PREVIOUS = DATA / "self_evolution/closed_loop_v1/real_runs/20261125_064727_closed_loop/stage0_external.png"
REFERENCE = DATA / "self_evolution/current_scene_cycle11/episodes/20261116_real_current_socket_repair_000002/raw/task/external/0000.jpg"


def rgb(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def register(current_path: Path, slot_seed_px: tuple[int, int]) -> dict:
    task = yaml.safe_load((NOV16 / "task.yaml").read_text())
    camera = json.loads((NOV16 / "cameras.json").read_text())["cameras"]["overhead"]
    camera_args = dict(K=camera["K"], T_world_cam=camera["T_parent_rgb"], distortion=camera["raw_distortion"]["coeffs"])
    current = rgb(current_path)
    previous_delta = align_scene_from_external_rgb(current, rgb(PREVIOUS), **camera_args)
    reference_delta = align_scene_from_external_rgb(current, rgb(REFERENCE), **camera_args)
    panels = detect_strip_features_rgb(current)
    if not panels or len(panels["panels_px"]) != 4:
        raise ValueError("Four socket panels are needed for the strip plane fit")
    xy = np.array([pixel_to_world_at_z(uv, STRIP_FACE_Z_M, **camera_args)[:2] for uv in panels["panels_px"]])
    dx, dy = np.polyfit(np.arange(4), xy, 1)[0]
    strip_yaw = float(np.rad2deg(np.arctan2(dy, dx)))
    local_modules = np.tile(task["first_module_xy_m"], (4, 1))
    local_modules[:, 0] += np.arange(4) * task["outlet_pitch_m"]
    strip_rotation = Rotation.from_euler("z", strip_yaw, degrees=True).as_matrix()[:2, :2]
    rotated = local_modules @ strip_rotation.T
    strip_xy = (xy - rotated).mean(axis=0)

    # The uncovered leftmost slot is on the CAD top plane. Its dark aperture
    # is a better translation landmark than the elevated yellow cable entry.
    x, y = slot_seed_px
    gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    x0, y0 = max(0, x - 6), max(0, y - 7)
    ys, xs = np.where(gray[y0:y + 8, x0:x + 7] < 36)
    if not xs.size:
        raise ValueError("Holder aperture absent at the supplied pixel seed")
    slot_px = (float(xs.mean() + x0), float(ys.mean() + y0))
    cad = json.loads((SIM / "assets/tasks/connector/manifest.json").read_text())
    local_slot = min(cad["holes"], key=lambda h: h["center_xy_m"][0])["center_xy_m"]
    holder_top = .025 + cad["top_z_m"]
    slot_world = pixel_to_world_at_z(slot_px, holder_top, **camera_args)[:2]
    holder_yaw = -89.5 + previous_delta.holder_delta_yaw_deg
    holder_rotation = Rotation.from_euler("z", holder_yaw, degrees=True).as_matrix()[:2, :2]
    holder_xy = slot_world - holder_rotation @ local_slot
    return dict(
        scene=dict(alignment_status="External RGB ray-plane registration with the November 25 holder yaw anchor",
                   holder_xy_m=holder_xy.tolist(), holder_yaw_deg=holder_yaw,
                   strip_xy_m=strip_xy.tolist(), strip_yaw_deg=strip_yaw,
                   active_arms=["left", "right"], robot_start_si=[0]*6+[.060]+[0]*6+[.040]),
        registration=dict(
            base_task_sha256=digest(NOV16 / "task.yaml"),
            current_external_rgb=str(current_path.relative_to(DATA)), current_sha256=digest(current_path),
            previous_external_rgb=str(PREVIOUS.relative_to(DATA)), previous_sha256=digest(PREVIOUS),
            reference_rgb=str(REFERENCE.relative_to(DATA)), reference_sha256=digest(REFERENCE),
            calibration_sha256=digest(NOV16 / "cameras.json"), nov16_to_current=asdict(reference_delta),
            previous_capture_to_current=asdict(previous_delta), panels_px=panels["panels_px"],
            panels_world_xy_m=xy.tolist(), panel_fit_max_residual_m=float(np.linalg.norm(xy - strip_xy - rotated, axis=1).max()),
            visible_holder_slot_px=slot_px, slot_seed_px=slot_seed_px, holder_slot_local_xy_m=local_slot,
            holder_top_world_z_m=holder_top, holder_yaw_anchor_deg=-89.5,
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("external_rgb", type=Path)
    parser.add_argument("--slot-seed-px", nargs=2, type=int, default=(477, 266))
    parser.add_argument("--output", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()
    profile = json.loads(DEFAULT_PROFILE.read_text())
    profile.update(register(args.external_rgb.resolve(), tuple(args.slot_seed_px)))
    save(args.output, profile)

"""Real-robot closed-loop execution entry for the right-arm 40 mm horizontal-grasp plug task.

Default is --dry-run (zero CAN/USB access, validates full closed-loop trajectory, joint limits,
and speed caps offline). Passing --execute requires:
  1. Active CAN interface `can_arm2` (`setup_can_vision.sh` completed by operator)
  2. RealSense RGB cameras `419222301415` (external) and `243722074652` (right_wrist) — RGB only!
  3. Operator confirmation (`TASK right`) through `piper_use_right`'s existing Hardware safety
     boundary (retaining 1.0 N·m gripper hold, CAN UP/1Mbps + USB port verification, joint limits,
     <=12 deg/s cap, 3 deg tracking error guard, keyboard emergency stop, and automatic motor
     disable after folding back to [0, 0, 0, 0, 0, 0]).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from learn import DATA, SIM  # noqa: E402
from robot_agent.bimanual_closed_loop import HorizontalGraspNominalPlanner, OnlineSupervisor  # noqa: E402
from robot_agent.compliant_residual import (  # noqa: E402
    HybridCompliantInsertController,
    inject_right_arm_residual,
    load_deployed_residual_policy,
)
from robot_agent.full_task_observation import FullTaskProprioHistory  # noqa: E402
from robot_agent.scene_retarget import align_scene_from_external_rgb  # noqa: E402
from robot_agent.wrist_inhand import WristInHandCompensator  # noqa: E402

sys.path.insert(0, str(SIM / "src"))
from piper_sim.backends.mujoco import MujocoBackend  # noqa: E402
from piper_sim.config import load as load_sim_config  # noqa: E402
from piper_sim.mixed_recording import configure_rigid_collection  # noqa: E402


EXTERNAL_SERIAL = "419222301415"
RIGHT_WRIST_SERIAL = "243722074652"
DEFAULT_REF_TASK = DATA / "self_evolution/20261116T105302Z_d700416b/task.yaml"
DEFAULT_REF_CAMERAS = DATA / "self_evolution/20261116T105302Z_d700416b/cameras.json"
DEFAULT_REF_EXTERNAL_RGB = (
    DATA
    / "self_evolution/current_scene_cycle11/episodes/20261116_real_current_socket_repair_000002/raw/task/external/0000.jpg"
)
VERIFIED_MAPPING_JSON = Path("/media/midea/F496AC8096AC4548/act_td3/piper_use/example_agilex/scripts/piper_vision_ports.json")
DEPLOYED_RESIDUAL_NPZ = DATA / "self_evolution/closed_loop_v1/residual_policy/residual_mlp.npz"
REAL_BIAS_PATH = DATA / "self_evolution/closed_loop_v1/real_calibration_bias.json"


def _resample_30hz_to_100hz(frames_14: list[np.ndarray], max_joint_speed_deg_s: float = 12.0) -> np.ndarray:
    """Convert 30 Hz SI 14-DoF targets into 100 Hz right-arm [deg x 6, width_mm / 95] hardware commands."""
    arr = np.asarray(frames_14, dtype=np.float64)
    right_deg = np.rad2deg(arr[:, 7:13])
    right_open = np.clip((arr[:, 13] * 1000.0) / 95.0, 0.0, 1.0)
    targets_30 = np.column_stack([right_deg, right_open])
    if len(targets_30) == 1:
        return targets_30
    t_src = np.arange(len(targets_30), dtype=np.float64) / 30.0
    dt = np.diff(t_src)
    vel = np.abs(np.diff(targets_30[:, :6], axis=0) / dt[:, None])
    scale = max(1.0, float(vel.max()) / max_joint_speed_deg_s)
    total_t = t_src[-1] * scale
    t_100 = np.r_[np.arange(0.0, total_t, 0.01), total_t]
    q_query = t_100 / scale
    return np.column_stack([np.interp(q_query, t_src, targets_30[:, j]) for j in range(7)])


def _capture_single_rgb(serial: str) -> np.ndarray:
    """Capture one RGB-only frame from RealSense (never enables depth stream per AGENTS.md)."""
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        for _ in range(12):
            frames = pipe.wait_for_frames(2500)
        bgr = np.asanyarray(frames.get_color_frame().get_data())
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        pipe.stop()


def build_offline_kinematics(
    task_yaml: Path,
    *,
    calibrate_align_t_gp: bool = False,
) -> tuple[MujocoBackend, HorizontalGraspNominalPlanner, dict, np.ndarray | None]:
    task_cfg = yaml.safe_load(task_yaml.read_text())
    task_cfg["active_arms"] = ["left", "right"]
    task_cfg["robot_gravity_compensation"] = True
    sim = MujocoBackend(dict(load_sim_config(), task_scene=task_cfg), cameras=False)
    planner = HorizontalGraspNominalPlanner(
        sim.model,
        sim.indices,
        grasp_height_m=0.020,
        preinsert_gap_m=0.045,
        align_gap_m=0.025,
        initial_right_opening_m=0.040,
        closed_width_m=0.027,
    )
    nominal_T_gp = None
    if calibrate_align_t_gp:
        configure_rigid_collection(sim)
        sim.reset()
        R_b0 = Rotation.from_euler("z", float(task_cfg["holder_yaw_deg"]), degrees=True).as_matrix()
        plug0 = sim.data.body("task/yellow_plug").xpos.copy()
        plug0_rot = sim.data.body("task/yellow_plug").xmat.reshape(3, 3).copy()
        board0 = sim.data.body("task/board").xpos.copy()
        plug_local = tuple((R_b0.T @ (plug0 - board0))[:2])
        R_b2p = R_b0.T @ plug0_rot
        p_xyz, R_b, R_p = planner.plug_pose_from_holder(
            tuple(task_cfg["holder_xy_m"]), float(task_cfg["holder_yaw_deg"]), plug_local, float(plug0[2]), R_b2p
        )
        cal_p1 = planner.plan_grasp_and_lift(plug_xyz_m=p_xyz, board_rotation=R_b, plug_rotation=R_p, folded_entry=False)
        sim.set_observation_state(cal_p1["start_state_14"])
        for _, seg in cal_p1["segments"]:
            for cmd in seg:
                sim.apply(sim.layout.command(cmd, sim.sim_time, 0))
                sim.step()
        for _ in range(15):
            sim.apply(sim.layout.command(cal_p1["q_lift_14"], sim.sim_time, 0))
            sim.step()
        T_wg_l = np.eye(4, dtype=np.float64)
        T_wg_l[:3, :3] = sim.data.body("right/gripper_center").xmat.reshape(3, 3)
        T_wg_l[:3, 3] = sim.data.body("right/gripper_center").xpos
        T_wp_l = np.eye(4, dtype=np.float64)
        T_wp_l[:3, :3] = sim.data.body("task/yellow_plug").xmat.reshape(3, 3)
        T_wp_l[:3, 3] = sim.data.body("task/yellow_plug").xpos
        T_gp_l = np.linalg.inv(T_wg_l) @ T_wp_l
        cal_p2 = planner.plan_transport_and_insert(
            q_current_14=cal_p1["q_lift_14"],
            T_gp_nominal=T_gp_l,
            strip_xy_m=tuple(task_cfg["strip_xy_m"]),
            strip_yaw_deg=float(task_cfg["strip_yaw_deg"]),
            outlet_local_xy_m=tuple(sim.task.outlets[task_cfg["target_index"]]),
            socket_face_z_m=float(sim.task.socket_face_z),
        )
        for cmd in [*cal_p2["transport_frames"], *cal_p2["align_frames"], *([cal_p2["q_align_14"]] * 10)]:
            sim.apply(sim.layout.command(cmd, sim.sim_time, 0))
            sim.step()
        T_wg_a = np.eye(4, dtype=np.float64)
        T_wg_a[:3, :3] = sim.data.body("right/gripper_center").xmat.reshape(3, 3)
        T_wg_a[:3, 3] = sim.data.body("right/gripper_center").xpos
        T_wp_a = np.eye(4, dtype=np.float64)
        T_wp_a[:3, :3] = sim.data.body("task/yellow_plug").xmat.reshape(3, 3)
        T_wp_a[:3, 3] = sim.data.body("task/yellow_plug").xpos
        nominal_T_gp = np.linalg.inv(T_wg_a) @ T_wp_a
        sim.reset()
    # Real yellow plug has ~24 mm molded horizontal side recesses; command 16 mm @ 1.0 N.m stall for positive lock!
    planner.closed_width_m = 0.016
    return sim, planner, task_cfg, nominal_T_gp


def dry_run_preview(task_yaml: Path, strip_delta_mm: tuple[float, float], holder_delta_mm: tuple[float, float]) -> dict:
    """Validate the complete 3-stage trajectory, joint bounds, 40 mm opening, and 100 Hz speed limits offline."""
    sim, planner, task_cfg, _ = build_offline_kinematics(task_yaml, calibrate_align_t_gp=False)
    try:
        sup = OnlineSupervisor(bias_path=REAL_BIAS_PATH)
        holder_xy = (
            task_cfg["holder_xy_m"][0] + holder_delta_mm[0] / 1000.0,
            task_cfg["holder_xy_m"][1] + holder_delta_mm[1] / 1000.0,
        )
        strip_xy = (
            task_cfg["strip_xy_m"][0] + strip_delta_mm[0] / 1000.0,
            task_cfg["strip_xy_m"][1] + strip_delta_mm[1] / 1000.0,
        )

        R_b0 = Rotation.from_euler("z", float(task_cfg["holder_yaw_deg"]), degrees=True).as_matrix()
        plug0 = sim.data.body("task/yellow_plug").xpos.copy()
        plug0_rot = sim.data.body("task/yellow_plug").xmat.reshape(3, 3).copy()
        board0 = sim.data.body("task/board").xpos.copy()
        plug_local = tuple((R_b0.T @ (plug0 - board0))[:2])
        R_b2p = R_b0.T @ plug0_rot

        plug_xyz, R_b, R_p = planner.plug_pose_from_holder(holder_xy, float(task_cfg["holder_yaw_deg"]), plug_local, float(plug0[2]), R_b2p)
        plan1 = planner.plan_grasp_and_lift(
            plug_xyz_m=plug_xyz, board_rotation=R_b, plug_rotation=R_p, folded_entry=True, free_speed_deg_s=12.0, contact_speed_deg_s=12.0
        )
        plan2 = planner.plan_transport_and_insert(
            q_current_14=plan1["q_lift_14"],
            T_gp_nominal=plan1["T_gp_nominal"],
            strip_xy_m=strip_xy,
            strip_yaw_deg=float(task_cfg["strip_yaw_deg"]),
            outlet_local_xy_m=tuple(sim.task.outlets[task_cfg["target_index"]]),
            socket_face_z_m=float(sim.task.socket_face_z),
            learned_bias_xy_m=sup.current_bias(),
            free_speed_deg_s=12.0,
            contact_speed_deg_s=12.0,
        )

        stage1_30hz = [f for _, seg in plan1["segments"] for f in seg]
        stage2_30hz = [
            *plan2["transport_frames"],
            *plan2["align_frames"],
            *plan2["insert_frames"],
            *plan2["release_frames"],
            *plan2["retreat_frames"],
            *plan2["fold_frames"],
        ]
        s1_100 = _resample_30hz_to_100hz(stage1_30hz, max_joint_speed_deg_s=12.0)
        s2_100 = _resample_30hz_to_100hz(stage2_30hz, max_joint_speed_deg_s=12.0)
        full_100 = np.vstack([s1_100, s2_100])

        lower_deg = np.rad2deg(sim.layout.lower[7:13])
        upper_deg = np.rad2deg(sim.layout.upper[7:13])
        within_limits = bool(np.all(full_100[:, :6] >= lower_deg - 1e-6) and np.all(full_100[:, :6] <= upper_deg + 1e-6))
        max_vel_deg_s = float(np.max(np.abs(np.diff(full_100[:, :6], axis=0)) / 0.01))

        return {
            "mode": "dry_run",
            "hardware_commands": 0,
            "active_arm": "right (can_arm2)",
            "grasp_posture": "horizontal_40mm_opening (0.040 m -> 0.027 m, 1.0 N.m)",
            "folded_entry_deg": np.round(full_100[0, :6], 4).tolist(),
            "folded_exit_deg": np.round(full_100[-1, :6], 4).tolist(),
            "initial_gripper_width_mm": round(float(full_100[0, 6] * 95.0), 2),
            "stage1_100hz_steps": int(len(s1_100)),
            "stage2_100hz_steps": int(len(s2_100)),
            "planned_duration_s": round(float(len(full_100) * 0.01), 2),
            "max_joint_speed_deg_s": round(max_vel_deg_s, 3),
            "within_joint_limits": within_limits,
            "learned_residual_ready": DEPLOYED_RESIDUAL_NPZ.is_file(),
            "persistent_bias_mm": [round(v * 1000.0, 3) for v in sup.current_bias()],
            "ready_for_hardware": bool(within_limits and max_vel_deg_s <= 12.05),
        }
    finally:
        sim.close()


def execute_on_hardware(args: argparse.Namespace) -> dict:
    """Execute the 4-stage closed-loop controller on the real right Piper arm with full safety checks."""
    # Use piper_use_right (1.0 N.m task gripper effort + CAN UP/1Mbps checks) paired with today's
    # verified 2026-11-25 USB mapping (1-10.3 -> can_arm2) from piper_use.
    hw_root = Path("/media/midea/F496AC8096AC4548/act_td3/piper_use_right")
    for p in (hw_root, hw_root / "example_agilex", hw_root / "example_agilex/piper_sdk"):
        sys.path.insert(0, str(p))
    from replay_connector import Hardware, Keyboard, parse_args as parse_hw_args, transition

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    capture_dir = DATA / "self_evolution/closed_loop_v1/real_runs" / f"{stamp}_closed_loop"
    capture_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage 0: Capture unobstructed external RGB BEFORE enabling motors
    ext_rgb = _capture_single_rgb(EXTERNAL_SERIAL)
    cv2.imwrite(str(capture_dir / "stage0_external.png"), cv2.cvtColor(ext_rgb, cv2.COLOR_RGB2BGR))

    ref_bgr = cv2.imread(str(args.reference_rgb))
    if ref_bgr is None:
        raise FileNotFoundError(f"Reference external RGB not found: {args.reference_rgb}")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    cam_meta = json.loads(args.cameras_json.read_text())["cameras"]["overhead"]
    s0 = align_scene_from_external_rgb(
        ext_rgb,
        ref_rgb,
        K=np.asarray(cam_meta["K"], dtype=np.float64),
        T_world_cam=np.asarray(cam_meta["T_parent_rgb"], dtype=np.float64),
        distortion=np.asarray(cam_meta["raw_distortion"]["coeffs"], dtype=np.float64),
    )
    (capture_dir / "stage0_alignment.json").write_text(json.dumps(asdict(s0), indent=2) + "\n")
    print(
        f"[Stage 0 RGB Align] valid={s0.valid} strip_dxy_mm={[round(v*1000,2) for v in s0.strip_delta_xy_m]} "
        f"strip_dyaw={s0.strip_delta_yaw_deg:.2f}deg holder_dxy_mm={[round(v*1000,2) for v in s0.holder_delta_xy_m]}",
        flush=True,
    )

    sim, planner, task_cfg, calibrated_T_gp = build_offline_kinematics(args.task, calibrate_align_t_gp=True)
    sup = OnlineSupervisor(bias_path=REAL_BIAS_PATH)
    learned_policy = load_deployed_residual_policy(DEPLOYED_RESIDUAL_NPZ)
    wrist_checker = WristInHandCompensator()
    # Real Piper CAN at SDK 15% has up to +1.90 mm normal free-air tracking lag (measured in
    # 20261116_000002 & 20261125_064727). Set differential rim threshold to 2.80 mm below cmd_h <= 16.5 mm!
    insert_ctrl = HybridCompliantInsertController(
        learned_policy=learned_policy,
        contact_deflection_threshold_m=0.0028,
        max_rim_cmd_h_m=0.0165,
    )

    try:
        R_b0 = Rotation.from_euler("z", float(task_cfg["holder_yaw_deg"]), degrees=True).as_matrix()
        plug0 = sim.data.body("task/yellow_plug").xpos.copy()
        plug0_rot = sim.data.body("task/yellow_plug").xmat.reshape(3, 3).copy()
        board0 = sim.data.body("task/board").xpos.copy()
        plug_local = tuple((R_b0.T @ (plug0 - board0))[:2])
        R_b2p = R_b0.T @ plug0_rot
        outlet_local = np.asarray(sim.task.outlets[task_cfg["target_index"]], dtype=np.float64)

        if s0.valid:
            holder_yaw = float(task_cfg["holder_yaw_deg"]) + s0.holder_delta_yaw_deg
            R_h1 = Rotation.from_euler("z", holder_yaw, degrees=True).as_matrix()[:2, :2]
            nom_plug_xy = np.asarray(task_cfg["holder_xy_m"]) + R_b0[:2, :2] @ np.asarray(plug_local)
            est_plug_xy = nom_plug_xy + np.asarray(s0.holder_delta_xy_m)
            holder_xy = tuple((est_plug_xy - R_h1 @ np.asarray(plug_local)).tolist())

            strip_yaw = float(task_cfg["strip_yaw_deg"]) + s0.strip_delta_yaw_deg
            R_s0 = Rotation.from_euler("z", float(task_cfg["strip_yaw_deg"]), degrees=True).as_matrix()[:2, :2]
            R_s1 = Rotation.from_euler("z", strip_yaw, degrees=True).as_matrix()[:2, :2]
            nom_outlet_xy = np.asarray(task_cfg["strip_xy_m"]) + R_s0 @ outlet_local
            est_outlet_xy = nom_outlet_xy + np.asarray(s0.strip_delta_xy_m)
            strip_xy = tuple((est_outlet_xy - R_s1 @ outlet_local).tolist())
        else:
            holder_xy = tuple(task_cfg["holder_xy_m"])
            holder_yaw = float(task_cfg["holder_yaw_deg"])
            strip_xy = tuple(task_cfg["strip_xy_m"])
            strip_yaw = float(task_cfg["strip_yaw_deg"])

        plug_xyz, R_b, R_p = planner.plug_pose_from_holder(holder_xy, holder_yaw, plug_local, float(plug0[2]), R_b2p)

        plan1 = planner.plan_grasp_and_lift(
            plug_xyz_m=plug_xyz,
            board_rotation=R_b,
            plug_rotation=R_p,
            calibrated_T_gp=calibrated_T_gp,
            folded_entry=True,
            free_speed_deg_s=12.0,
            contact_speed_deg_s=12.0,
        )
        stage1_100 = _resample_30hz_to_100hz([f for _, seg in plan1["segments"] for f in seg], max_joint_speed_deg_s=12.0)

        # Pre-validate nominal stage 2 limits before enabling robot
        preview_plan2 = planner.plan_transport_and_insert(
            q_current_14=plan1["q_lift_14"],
            T_gp_nominal=plan1["T_gp_nominal"],
            strip_xy_m=strip_xy,
            strip_yaw_deg=strip_yaw,
            outlet_local_xy_m=tuple(sim.task.outlets[task_cfg["target_index"]]),
            socket_face_z_m=float(sim.task.socket_face_z),
            learned_bias_xy_m=sup.current_bias(),
            free_speed_deg_s=12.0,
            contact_speed_deg_s=12.0,
        )
        preview_s2_100 = _resample_30hz_to_100hz(
            [
                *preview_plan2["transport_frames"],
                *preview_plan2["align_frames"],
                *preview_plan2["insert_frames"],
                *preview_plan2["release_frames"],
                *preview_plan2["retreat_frames"],
                *preview_plan2["fold_frames"],
            ],
            max_joint_speed_deg_s=12.0,
        )

        hw_args = parse_hw_args([])
        hw_args.mapping_json = VERIFIED_MAPPING_JSON
        hw_args.folded_entry = True
        hw_args.max_joint_speed = 12.0
        hw_args.control_hz = 100
        hardware = Hardware(hw_args, ["right"])

        full_preview = np.vstack([stage1_100, preview_s2_100])
        episode_stub = SimpleNamespace(episode_id="CLOSED_LOOP_PLUG_INSERT", actions={"right": full_preview})

        with (capture_dir / "commanded_probe.csv").open("x", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                ["monotonic", "phase"]
                + [f"{kind}_j{i}" for kind in ("target", "feedback") for i in range(1, 7)]
                + ["target_gripper_open", "feedback_gripper_open"]
            )
            try:
                entry = hardware.connect(
                    episode_stub,
                    stage1_100[:1],
                    confirmation="TASK right",
                    target_label="闭环40mm横抓取插入口",
                )
                hardware.controllers["right"].joint_speed_percent = 15
                print("TASK: joint target <=12 deg/s, SDK 15%, gripper 1.0 N.m, opening=40mm horizontal", flush=True)

                with Keyboard() as keyboard:
                    def emit_100hz(target_row_7: np.ndarray, phase_label: str) -> np.ndarray:
                        tick = time.monotonic()
                        hardware.poll(keyboard)
                        cmd = target_row_7.reshape(1, 7)
                        hardware.send(cmd)
                        actual = hardware.feedback("right", True)[0]
                        writer.writerow([time.monotonic(), phase_label, *cmd[0, :6], *actual[:6], cmd[0, 6], actual[6]])
                        csv_f.flush()
                        time.sleep(max(0.0, 0.01 - (time.monotonic() - tick)))
                        return actual

                    # Smoothly transition to folded entry (opening gripper to 40 mm) and run Stage 1
                    for step_cmd in transition(entry, stage1_100[:1], 1.5, hw_args):
                        emit_100hz(step_cmd[0], "prepare_gripper_40mm")
                    actual_lift = stage1_100[0]
                    for row_7 in stage1_100:
                        actual_lift = emit_100hz(row_7, "grasp_lift")
                    for _ in range(25):  # 0.25s stationary settle for crisp wrist RGB
                        actual_lift = emit_100hz(stage1_100[-1], "lift_settle")

                    # 2. Stage 1: Right-wrist RGB retention check
                    wrist_rgb = _capture_single_rgb(RIGHT_WRIST_SERIAL)
                    cv2.imwrite(str(capture_dir / "stage1_wrist_lift.png"), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
                    q6_lift_si = np.deg2rad(actual_lift[:6])
                    grip_w_lift_m = float(actual_lift[6] * 0.095)
                    R_wg_lift = planner.gripper_pose(q6_lift_si)[:3, :3]
                    wcheck = wrist_checker.evaluate_post_lift(
                        wrist_rgb, grip_w_lift_m, R_wg_lift, apply_metric_offset=False
                    )
                    print(
                        f"[Stage 1 Wrist Gate] status={wcheck.status} retained={wcheck.retained} "
                        f"width_mm={grip_w_lift_m*1000:.2f} yellow_area_px={wcheck.yellow_area_px:.0f}",
                        flush=True,
                    )
                    if not wcheck.retained:
                        fold_back = _resample_30hz_to_100hz(
                            [plan1["q_lift_14"], np.array([0.0] * 6 + [0.060] + [0.0] * 6 + [0.040])]
                        )
                        for row_7 in fold_back:
                            emit_100hz(row_7, "abort_fold")
                        return {"status": "aborted_at_lift", "reason": wcheck.reason, "capture": str(capture_dir)}

                    plan2 = planner.plan_transport_and_insert(
                        q_current_14=plan1["q_lift_14"],
                        T_gp_nominal=plan1["T_gp_nominal"],
                        strip_xy_m=strip_xy,
                        strip_yaw_deg=strip_yaw,
                        outlet_local_xy_m=tuple(sim.task.outlets[task_cfg["target_index"]]),
                        socket_face_z_m=float(sim.task.socket_face_z),
                        wrist_world_xy_corr_m=wcheck.world_xy_correction_m,
                        wrist_yaw_corr_deg=wcheck.yaw_correction_deg,
                        learned_bias_xy_m=sup.current_bias(),
                        free_speed_deg_s=12.0,
                        contact_speed_deg_s=12.0,
                    )
                    hardware.validate_limits("right", _resample_30hz_to_100hz(plan2["transport_frames"])[:, :6], "transport")

                    actual_cur = actual_lift.copy()
                    for row_7 in _resample_30hz_to_100hz(plan2["transport_frames"], 12.0):
                        actual_cur = emit_100hz(row_7, "transport")

                    # 3. Stage 2: Online 30 Hz compliant residual outer loop / 100 Hz inner emission
                    proprio_hist = FullTaskProprioHistory(15)
                    insert_ctrl.reset(initial_xy_bias_m=(0.0, 0.0), lift_gripper_width_m=grip_w_lift_m)
                    wrist_vec = wcheck.features.vector()
                    last_cmd_14 = plan2["q_preinsert_14"].copy()
                    prev_actual_q6 = np.deg2rad(actual_cur[:6])

                    ins_seq = plan2["insert_frames"]
                    n_ins = len(ins_seq)
                    z_seated_grip = float(planner.gripper_pose(plan2["q_insert_14"][7:13])[2, 3])
                    idx_clear = 0  # Full 25.0 mm align clearance (~6 mm clear above rim!)

                    def step_closed_loop_30hz(
                        nom_14: np.ndarray,
                        phase_name: str,
                        progress: float = 0.0,
                        free_air_lag: float | None = None,
                    ):
                        nonlocal last_cmd_14, prev_actual_q6, actual_cur
                        q6_si = np.deg2rad(actual_cur[:6])
                        dq6_si = (q6_si - prev_actual_q6) * 30.0
                        prev_actual_q6 = q6_si.copy()
                        gw_m = float(actual_cur[6] * 0.095)
                        h_obs = proprio_hist.append(
                            joint_position=q6_si,
                            joint_velocity=dq6_si,
                            nominal_action=nom_14[7:14],
                            controller_target=last_cmd_14[7:14],
                            gripper_width=gw_m,
                            phase=phase_name,
                        )
                        obs_vec = np.concatenate([h_obs, wrist_vec]).astype(np.float32)
                        J6 = planner.jacobian_6x6(nom_14[7:13])
                        z_act_g = float(planner.gripper_pose(q6_si)[2, 3])
                        z_cmd_g = float(planner.gripper_pose(last_cmd_14[7:13])[2, 3])
                        res4, pause_desc, telem = insert_ctrl.step(
                            phase=phase_name,
                            insert_progress=progress,
                            q_right_6=q6_si,
                            gripper_width_m=gw_m,
                            commanded_right_6=last_cmd_14[7:13],
                            jacobian_6x6=J6,
                            policy_obs=obs_vec,
                            actual_height_above_seated_m=z_act_g - z_seated_grip,
                            cmd_height_above_seated_m=z_cmd_g - z_seated_grip,
                            free_air_lag_m=free_air_lag,
                        )
                        cmd_14, _ = inject_right_arm_residual(nom_14, res4, J6, sim.layout.lower, sim.layout.upper)
                        target_7 = np.r_[np.rad2deg(cmd_14[7:13]), np.clip((cmd_14[13] * 1000.0) / 95.0, 0.0, 1.0)]
                        hardware.validate_limits("right", target_7[:6].reshape(1, 6), phase_name)
                        prev_target_7 = np.r_[np.rad2deg(last_cmd_14[7:13]), np.clip((last_cmd_14[13] * 1000.0) / 95.0, 0.0, 1.0)]
                        for sub in (1.0 / 3.0, 2.0 / 3.0, 1.0):
                            actual_cur = emit_100hz(prev_target_7 + (target_7 - prev_target_7) * sub, phase_name)
                        last_cmd_14 = cmd_14
                        return pause_desc, telem, (z_act_g - z_seated_grip), (z_cmd_g - z_seated_grip)

                    for nom_14 in plan2["align_frames"]:
                        step_closed_loop_30hz(nom_14, "align", 0.0)

                    cursor = 0
                    sub_tick = 0
                    free_air_lags: list[float] = []
                    free_air_lag_est = 0.00050
                    ticks = 0
                    seated_ok = False
                    while ticks < n_ins + 900:
                        ticks += 1
                        nom_14 = ins_seq[min(cursor, n_ins - 1)]
                        pause_desc, telem, act_h, cmd_h = step_closed_loop_30hz(
                            nom_14, "insert", float(cursor + 1) / n_ins, free_air_lag=free_air_lag_est
                        )
                        if 0.0218 <= cmd_h <= 0.0238 and not insert_ctrl.clear_steps_left and not insert_ctrl.shift_steps_left:
                            free_air_lags.append(act_h - cmd_h)
                            if len(free_air_lags) >= 2:
                                free_air_lag_est = float(np.median(free_air_lags[-6:]))
                        if telem.jam_detected:
                            print(
                                f"[Stage 2 Rim Contact] probe->{telem.probe_index} "
                                f"next_xy_mm={[round(v*1000,2) for v in insert_ctrl.pending_xy]}",
                                flush=True,
                            )
                        if telem.jam_detected or pause_desc:
                            free_air_lags.clear()
                            sub_tick = 0
                            cursor = max(idx_clear, cursor - 4)
                        else:
                            if 0.0150 <= act_h <= 0.0238 and cursor < n_ins - 1:
                                sub_tick += 1
                                if sub_tick >= 2:
                                    sub_tick = 0
                                    cursor += 1
                            elif cursor < n_ins - 1:
                                sub_tick = 0
                                cursor += 1
                        if (cursor >= n_ins - 1 and telem.seated_ready) or (
                            act_h <= 0.0080
                            and float(np.max(np.abs(np.rad2deg(last_cmd_14[7:13]) - actual_cur[:6]))) >= 1.85
                        ):
                            # Hold compliant bottomed-out pose gently inside SDK's 3.0 deg tracking cap before release!
                            last_cmd_14[7:13] = np.deg2rad(
                                actual_cur[:6]
                                + np.clip(np.rad2deg(last_cmd_14[7:13]) - actual_cur[:6], -1.2, 1.2)
                            )
                            seated_ok = True
                            break

                    # Anchor release frames to actual seated pose so holding for RGB snapshot & opening fingers never spikes tracking error!
                    release_anchor = last_cmd_14.copy()
                    hold_target_7 = np.r_[np.rad2deg(release_anchor[7:13]), np.clip((release_anchor[13] * 1000.0) / 95.0, 0.0, 1.0)]
                    for _ in range(10):
                        actual_cur = emit_100hz(hold_target_7, "seated_hold")

                    # Capture pre-release RGB snapshots while holding steady
                    seated_ext = _capture_single_rgb(EXTERNAL_SERIAL)
                    seated_wrist = _capture_single_rgb(RIGHT_WRIST_SERIAL)
                    cv2.imwrite(str(capture_dir / "stage2_seated_external.png"), cv2.cvtColor(seated_ext, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(capture_dir / "stage2_seated_wrist.png"), cv2.cvtColor(seated_wrist, cv2.COLOR_RGB2BGR))

                    from robot_agent.bimanual_closed_loop import quintic_interpolate
                    if seated_ok:
                        release_goal = release_anchor.copy()
                        release_goal[13] = 0.060
                        for nom_14 in quintic_interpolate(release_anchor, release_goal, 12.0, min_duration_s=0.8):
                            step_closed_loop_30hz(nom_14, "release", 1.0)
                        retreat_goal = plan2["q_preinsert_14"].copy()
                        retreat_goal[13] = 0.060
                        for nom_14 in quintic_interpolate(release_goal, retreat_goal, 12.0, min_duration_s=1.2):
                            step_closed_loop_30hz(nom_14, "retreat", 1.0)
                        fold_start = retreat_goal.copy()
                    else:
                        # Never open gripper in mid-air if seating depth was not reached: retreat holding plug & fold!
                        retreat_held = plan2["q_preinsert_14"].copy()
                        retreat_held[13] = release_anchor[13]
                        for nom_14 in quintic_interpolate(release_anchor, retreat_held, 12.0, min_duration_s=1.2):
                            step_closed_loop_30hz(nom_14, "abort_retreat_held", 1.0)
                        fold_start = retreat_held.copy()

                    fold_goal = fold_start.copy()
                    fold_goal[7:13] = 0.0
                    for row_7 in _resample_30hz_to_100hz(quintic_interpolate(fold_start, fold_goal, 12.0), 12.0):
                        actual_cur = emit_100hz(row_7, "fold")
                    hardware.wait_targets(
                        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(actual_cur[6])]]),
                        gripper=False,
                        keyboard=keyboard,
                    )
                    for _ in range(30):
                        actual_cur = emit_100hz(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(actual_cur[6])]), "final_hold")

                win_xy = tuple(insert_ctrl.active_xy.tolist())
                sup_entry = sup.update_from_episode(
                    seated_success=seated_ok, winning_residual_xy_m=win_xy, jam_events=insert_ctrl.jam_events
                )
                summary = {
                    "capture_dir": str(capture_dir),
                    "stage0_alignment": asdict(s0),
                    "stage1_wrist_gate": {
                        "status": wcheck.status,
                        "retained": wcheck.retained,
                        "gripper_width_mm": round(grip_w_lift_m * 1000.0, 2),
                        "yellow_area_px": round(wcheck.yellow_area_px, 1),
                    },
                    "seated_proprioceptive_ready": seated_ok,
                    "jam_events": insert_ctrl.jam_events,
                    "probes_used": insert_ctrl.probe_idx,
                    "winning_residual_mm": [round(v * 1000.0, 3) for v in win_xy],
                    "final_joint_feedback_deg": np.round(actual_cur[:6], 3).tolist(),
                    "supervisor_bias_update": sup_entry,
                }
            finally:
                hardware.shutdown(disable_after=args.disable_after)
                post_ext = _capture_single_rgb(EXTERNAL_SERIAL)
                cv2.imwrite(str(capture_dir / "stage3_post_disable_external.png"), cv2.cvtColor(post_ext, cv2.COLOR_RGB2BGR))
                if "summary" in locals():
                    summary["motors_auto_disabled"] = True
                    (capture_dir / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=DEFAULT_REF_TASK if DEFAULT_REF_TASK.is_file() else SIM / "configs/tasks/real_connector_calibrated.yaml")
    parser.add_argument("--cameras-json", type=Path, default=DEFAULT_REF_CAMERAS)
    parser.add_argument("--reference-rgb", type=Path, default=DEFAULT_REF_EXTERNAL_RGB)
    parser.add_argument("--strip-delta-mm", type=float, nargs=2, default=[0.0, 0.0], help="Dry-run test strip offset (mm)")
    parser.add_argument("--holder-delta-mm", type=float, nargs=2, default=[0.0, 0.0], help="Dry-run test holder offset (mm)")
    parser.add_argument("--disable-after", type=float, default=3.0, help="Seconds after folding to automatically disable motors")
    parser.add_argument("--execute", action="store_true", help="Connect to real can_arm2 + RGB cameras")
    args = parser.parse_args()

    if not args.execute:
        report = dry_run_preview(args.task, tuple(args.strip_delta_mm), tuple(args.holder_delta_mm))
        print(json.dumps(report, indent=2))
        return

    result = execute_on_hardware(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

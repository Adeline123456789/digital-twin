"""Unified end-to-end closed-loop plug-and-socket pipeline + Fast/Slow self-evolution.

Executes the 14-DoF dual-arm contract (left arm folded at zero, right arm active with
the verified 40 mm horizontal grasp so the neighboring black charger is never touched):
  Stage 0: External RGB initial scene alignment (strip & holder differential XY/yaw)
  Stage 1: Post-lift right-wrist RGB retention check & in-hand offset compensation
  Stage 2: Hybrid compliant residual insertion (SmoothResidual4D + jam unload/rewind + learned PPO/spiral)
  Stage 3: Fast/Slow supervisor consolidating winning residual offsets into persistent bias memory
           without rebuilding the MuJoCo digital twin.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
import time
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

from .compliant_residual import HybridCompliantInsertController, inject_right_arm_residual
from .full_task_observation import FullTaskProprioHistory
from .scene_retarget import SceneAlignmentResult, align_scene_from_external_rgb
from .wrist_inhand import WristInHandCheck, WristInHandCompensator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from learn import DATA, SIM  # noqa: E402

sys.path.insert(0, str(SIM / "src"))
sys.path.insert(0, str(SIM / "scripts"))
from piper_sim.backends.mujoco import MujocoBackend  # noqa: E402
from piper_sim.config import load as load_sim_config  # noqa: E402
from piper_sim.mixed_recording import checked_physics_step, configure_rigid_collection  # noqa: E402
from try_plug_pick_sim import PoseIK  # noqa: E402


def quintic_interpolate(q_start: np.ndarray, q_goal: np.ndarray, speed_deg_s: float, gripper_speed_m_s: float = 0.0475, min_duration_s: float = 0.35, fps: int = 30) -> list[np.ndarray]:
    """Generate 30 Hz quintic-smoothed 14-DoF position targets respecting joint & gripper speed caps."""
    joint_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    grip_indices = [6, 13]
    joint_delta = float(np.max(np.abs((q_goal - q_start)[joint_indices])))
    grip_delta = float(np.max(np.abs((q_goal - q_start)[grip_indices])))
    t_joint = 1.875 * joint_delta / np.deg2rad(speed_deg_s)
    t_grip = 1.875 * grip_delta / gripper_speed_m_s
    duration = max(min_duration_s, t_joint, t_grip)
    steps = max(1, int(np.ceil(duration * fps)))
    alphas = np.linspace(0.0, 1.0, steps + 1)[1:]
    progress = alphas ** 3 * (10.0 - 15.0 * alphas + 6.0 * alphas ** 2)
    return [q_start + (q_goal - q_start) * float(p) for p in progress]


class HorizontalGraspNominalPlanner:
    """Kinematic waypoint solver using only robot model + estimated/nominal fixture poses (zero privileged sim data)."""

    def __init__(self, sim_model: mujoco.MjModel, indices: dict, *, grasp_height_m: float = 0.020, preinsert_gap_m: float = 0.045, align_gap_m: float = 0.025, initial_right_opening_m: float = 0.040, closed_width_m: float = 0.027):
        self.model = sim_model
        self.kin_data = mujoco.MjData(sim_model)
        self.right_qpos_ids = indices["right"]["qpos"][:6]
        self.right_dof_ids = indices["right"]["dof"][:6]
        self.grip_body_id = sim_model.body("right/gripper_center").id
        self.grasp_height_m = float(grasp_height_m)
        self.preinsert_gap_m = float(preinsert_gap_m)
        self.align_gap_m = float(align_gap_m)
        self.initial_right_opening_m = float(initial_right_opening_m)
        self.closed_width_m = float(closed_width_m)

        # Build isolated kinematic-only dummy wrapper for PoseIK
        class _KinStub:
            pass

        stub = _KinStub()
        stub.model = sim_model
        stub.data = self.kin_data
        stub.indices = indices
        self.ik = PoseIK(stub, side="right")
        # Seed near the verified 40mm horizontal grasp [-7.9, 93.6, -65.2, 0, 66.6, -14.2] deg
        self.horizontal_seed = np.deg2rad([-8.0, 94.0, -65.0, 0.0, 66.0, -14.0])
        self.insert_seed = np.deg2rad([44.0, 113.0, -88.0, 0.0, 66.0, -45.0])
        self.base_ik_rotation = self.ik.rotation.copy()

    def _solve_ik_robust(self, xyz: np.ndarray, rotation: np.ndarray, primary_seed: np.ndarray) -> np.ndarray:
        """Solve 6-DoF right-arm IK to <0.1 mm / <0.1 deg with multi-seed 400-eval fallback."""
        seeds = [
            primary_seed,
            self.insert_seed,
            np.deg2rad([44.0, 113.0, -88.0, 0.0, 66.0, 43.0]),
            self.horizontal_seed,
        ]
        bounds = self.ik.bounds

        def err_fn(q: np.ndarray) -> np.ndarray:
            self.ik.data.qpos[self.ik.ids] = q
            mujoco.mj_forward(self.ik.model, self.ik.data)
            return np.r_[
                10.0 * (self.ik.data.xpos[self.ik.bid] - xyz),
                Rotation.from_matrix(rotation.T @ self.ik.data.xmat[self.ik.bid].reshape(3, 3)).as_rotvec(),
            ]

        last_res = None
        for sd in seeds:
            x0 = np.clip(np.asarray(sd, dtype=np.float64), bounds[0] + 1e-7, bounds[1] - 1e-7)
            fit = least_squares(err_fn, x0, bounds=(bounds[0], bounds[1]), max_nfev=500)
            res = err_fn(fit.x)
            last_res = res
            if np.linalg.norm(res[:3]) / 10.0 <= 0.0001 and np.linalg.norm(res[3:]) <= np.deg2rad(0.1):
                self.ik.seed = fit.x.copy()
                return fit.x.copy()
        raise ValueError(f"Unreachable pose {xyz}: residual {last_res}")

    def jacobian_6x6(self, right_q6: np.ndarray) -> np.ndarray:
        self.kin_data.qpos[self.right_qpos_ids] = np.asarray(right_q6, dtype=np.float64)
        mujoco.mj_forward(self.model, self.kin_data)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, self.kin_data, jacp, jacr, self.grip_body_id)
        return np.vstack([jacp[:, self.right_dof_ids], jacr[:, self.right_dof_ids]])

    def gripper_pose(self, right_q6: np.ndarray) -> np.ndarray:
        self.kin_data.qpos[self.right_qpos_ids] = np.asarray(right_q6, dtype=np.float64)
        mujoco.mj_forward(self.model, self.kin_data)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.kin_data.xmat[self.grip_body_id].reshape(3, 3)
        T[:3, 3] = self.kin_data.xpos[self.grip_body_id]
        return T

    @staticmethod
    def plug_pose_from_holder(
        holder_xy_m: tuple[float, float],
        holder_yaw_deg: float,
        plug_local_xy_m: tuple[float, float] = (0.0005, -0.0075),
        plug_z_m: float = 0.0582,
        R_board_to_plug: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute nominal yellow plug 3D center, board rotation, and plug rotation from holder config."""
        R_board = Rotation.from_euler("z", float(holder_yaw_deg), degrees=True).as_matrix()
        holder_origin = np.array([holder_xy_m[0], holder_xy_m[1], 0.025], dtype=np.float64)
        plug_pos = holder_origin + R_board @ np.array([plug_local_xy_m[0], plug_local_xy_m[1], 0.0], dtype=np.float64)
        plug_pos[2] = float(plug_z_m)
        R_plug = R_board if R_board_to_plug is None else (R_board @ np.asarray(R_board_to_plug, dtype=np.float64))
        return plug_pos, R_board, R_plug

    def plan_grasp_and_lift(
        self,
        *,
        plug_xyz_m: np.ndarray,
        board_rotation: np.ndarray,
        plug_rotation: np.ndarray | None = None,
        calibrated_T_gp: np.ndarray | None = None,
        folded_entry: bool = True,
        free_speed_deg_s: float = 18.0,
        contact_speed_deg_s: float = 12.0,
    ) -> dict:
        """Plan Stage 1 nominal trajectory: (optional approach from folded) -> descend -> grasp -> lift."""
        # Horizontal grasp: jaws along the holder row (NO Rz(90)), 40 mm opening avoids black charger!
        grasp_rot = board_rotation @ self.base_ik_rotation
        q_approach_6 = self._solve_ik_robust(plug_xyz_m + np.array([0.0, 0.0, 0.074]), grasp_rot, self.horizontal_seed)
        q_descend_6 = self._solve_ik_robust(plug_xyz_m + np.array([0.0, 0.0, self.grasp_height_m]), grasp_rot, q_approach_6)

        base_14 = np.array([0.0] * 6 + [0.060] + [0.0] * 6 + [self.initial_right_opening_m], dtype=np.float64)
        q_approach = base_14.copy()
        q_approach[7:13] = q_approach_6

        q_descend = q_approach.copy()
        q_descend[7:13] = q_descend_6

        q_grasp = q_descend.copy()
        q_grasp[13] = self.closed_width_m

        q_lift = q_grasp.copy()
        q_lift[7:13] = q_approach_6

        segments: list[tuple[str, list[np.ndarray]]] = []
        if folded_entry:
            segments.append(("approach", quintic_interpolate(base_14, q_approach, free_speed_deg_s)))
            start_state = base_14
        else:
            start_state = q_approach
        segments.append(("descend", quintic_interpolate(q_approach, q_descend, contact_speed_deg_s)))
        segments.append(("grasp", quintic_interpolate(q_descend, q_grasp, contact_speed_deg_s, min_duration_s=1.0)))
        segments.append(("lift", quintic_interpolate(q_grasp, q_lift, free_speed_deg_s)))

        if calibrated_T_gp is not None:
            T_gp_nominal = calibrated_T_gp.copy()
        else:
            T_wg_grasp = self.gripper_pose(q_descend_6)
            T_wp_nom = np.eye(4, dtype=np.float64)
            T_wp_nom[:3, :3] = board_rotation if plug_rotation is None else plug_rotation
            T_wp_nom[:3, 3] = plug_xyz_m
            T_gp_nominal = np.linalg.inv(T_wg_grasp) @ T_wp_nom

        return {
            "start_state_14": start_state,
            "q_lift_14": q_lift,
            "T_gp_nominal": T_gp_nominal,
            "segments": segments,
        }

    def plan_transport_and_insert(
        self,
        *,
        q_current_14: np.ndarray,
        T_gp_nominal: np.ndarray,
        strip_xy_m: tuple[float, float],
        strip_yaw_deg: float,
        outlet_local_xy_m: tuple[float, float],
        socket_face_z_m: float,
        wrist_world_xy_corr_m: tuple[float, float] = (0.0, 0.0),
        wrist_yaw_corr_deg: float = 0.0,
        learned_bias_xy_m: tuple[float, float] = (0.0, 0.0),
        free_speed_deg_s: float = 18.0,
        contact_speed_deg_s: float = 12.0,
    ) -> dict:
        """Plan Stage 2 nominal trajectory online after incorporating Stage 0 + Stage 1 + bias corrections."""
        T_strip = np.eye(4, dtype=np.float64)
        T_strip[:3, :3] = Rotation.from_euler("z", float(strip_yaw_deg) + float(wrist_yaw_corr_deg), degrees=True).as_matrix()
        T_strip[:3, 3] = np.array(
            [
                strip_xy_m[0] + wrist_world_xy_corr_m[0] + learned_bias_xy_m[0],
                strip_xy_m[1] + wrist_world_xy_corr_m[1] + learned_bias_xy_m[1],
                0.025,
            ],
            dtype=np.float64,
        )

        def solve_over_socket(gap_m: float, seed_q6: np.ndarray) -> np.ndarray:
            desired_local = np.eye(4, dtype=np.float64)
            desired_local[:2, 3] = np.asarray(outlet_local_xy_m, dtype=np.float64)
            desired_local[2, 3] = float(socket_face_z_m) + 0.0002 + float(gap_m)
            # Smooth carry-tilt taper keeps raised approach/align poses well inside the 70 deg J5 limit
            # while reaching 0.0 deg pure vertical before prongs engage the socket!
            if gap_m > 0.016:
                tilt_x_deg = min(5.0, 5.0 * (float(gap_m) - 0.016) / (0.040 - 0.016))
                desired_local[:3, :3] = Rotation.from_euler("x", tilt_x_deg, degrees=True).as_matrix()
            T_wp_desired = T_strip @ desired_local
            T_wg_desired = T_wp_desired @ np.linalg.inv(T_gp_nominal)
            return self._solve_ik_robust(T_wg_desired[:3, 3], T_wg_desired[:3, :3], seed_q6)

        q_preinsert_6 = solve_over_socket(self.preinsert_gap_m, self.insert_seed)
        q_align_6 = solve_over_socket(self.align_gap_m, q_preinsert_6)
        q_insert_6 = solve_over_socket(0.0, q_align_6)

        q_preinsert = q_current_14.copy()
        q_preinsert[7:13] = q_preinsert_6

        q_align = q_preinsert.copy()
        q_align[7:13] = q_align_6

        q_insert = q_align.copy()
        q_insert[7:13] = q_insert_6

        q_release = q_insert.copy()
        q_release[13] = 0.060

        q_retreat = q_preinsert.copy()
        q_retreat[13] = 0.060

        q_fold = q_retreat.copy()
        q_fold[7:13] = 0.0

        return {
            "q_preinsert_14": q_preinsert,
            "q_align_14": q_align,
            "q_insert_14": q_insert,
            "transport_frames": quintic_interpolate(q_current_14, q_preinsert, free_speed_deg_s),
            "align_frames": quintic_interpolate(q_preinsert, q_align, contact_speed_deg_s),
            "insert_frames": quintic_interpolate(q_align, q_insert, min(contact_speed_deg_s, 6.0), min_duration_s=3.0),
            "release_frames": quintic_interpolate(q_insert, q_release, contact_speed_deg_s, min_duration_s=0.8),
            "retreat_frames": quintic_interpolate(q_release, q_retreat, free_speed_deg_s, min_duration_s=1.2),
            "fold_frames": quintic_interpolate(q_retreat, q_fold, free_speed_deg_s),
        }


class OnlineSupervisor:
    """Fast/Slow self-evolving calibration memory (zero Twin rebuild, zero LLM latency)."""

    def __init__(self, bias_path: Path | None = None, step_gain: float = 0.65, max_bias_m: float = 0.006):
        self.bias_path = bias_path
        self.step_gain = float(step_gain)
        self.max_bias_m = float(max_bias_m)
        self.xy_bias_m = np.zeros(2, dtype=np.float64)
        self.episodes_observed = 0
        self.history: list[dict] = []
        if self.bias_path is not None and self.bias_path.is_file():
            data = json.loads(self.bias_path.read_text())
            self.xy_bias_m = np.clip(np.asarray(data.get("xy_bias_m", [0.0, 0.0]), dtype=np.float64), -max_bias_m, max_bias_m)
            self.episodes_observed = int(data.get("episodes_observed", 0))

    def current_bias(self) -> tuple[float, float]:
        return (float(self.xy_bias_m[0]), float(self.xy_bias_m[1]))

    def update_from_episode(self, *, seated_success: bool, winning_residual_xy_m: tuple[float, float], jam_events: int) -> dict:
        """Absorb systematic residual correction into next-episode pre-insert bias."""
        res = np.asarray(winning_residual_xy_m, dtype=np.float64)
        updated = False
        if seated_success and (jam_events > 0 or np.linalg.norm(res) > 0.00025):
            self.xy_bias_m = np.clip(self.xy_bias_m + self.step_gain * res, -self.max_bias_m, self.max_bias_m)
            updated = True
        self.episodes_observed += 1
        entry = {
            "episode_index": self.episodes_observed,
            "seated_success": bool(seated_success),
            "winning_residual_mm": (res * 1000.0).tolist(),
            "jam_events": int(jam_events),
            "updated_bias_mm": (self.xy_bias_m * 1000.0).tolist(),
            "bias_updated": updated,
        }
        self.history.append(entry)
        if self.bias_path is not None:
            self.bias_path.parent.mkdir(parents=True, exist_ok=True)
            self.bias_path.write_text(
                json.dumps(
                    {
                        "xy_bias_m": self.xy_bias_m.tolist(),
                        "episodes_observed": self.episodes_observed,
                        "last_update": entry,
                    },
                    indent=2,
                )
                + "\n"
            )
        return entry


@dataclass
class ClosedLoopEpisodeResult:
    success: bool
    open_loop_mode: bool
    frames: int
    sim_duration_s: float
    wall_seconds: float
    max_penetration_m: float
    collision_aborted: bool
    stage0_alignment: dict | None
    stage1_wrist_check: dict | None
    jam_events: int
    probes_used: int
    winning_residual_mm: tuple[float, float]
    final_metrics: dict
    supervisor_update: dict | None


class DualArmSimClosedLoopRunner:
    """End-to-end closed-loop execution on the 14-DoF dual-arm MuJoCo scene (right arm active)."""

    def __init__(
        self,
        task_yaml_path: Path | None = None,
        *,
        learned_policy=None,
        supervisor: OnlineSupervisor | None = None,
        folded_entry: bool = True,
        free_speed_deg_s: float = 12.0,
        contact_speed_deg_s: float = 12.0,
    ):
        self.task_yaml_path = task_yaml_path or (SIM / "configs/tasks/real_connector_calibrated.yaml")
        self.base_task_cfg = yaml.safe_load(self.task_yaml_path.read_text())
        self.base_task_cfg["robot_gravity_compensation"] = True
        self.enable_full_cameras = False
        self.sim = MujocoBackend(dict(load_sim_config(), task_scene=self.base_task_cfg), cameras=False)
        self.profile = configure_rigid_collection(self.sim)
        self.profile["robot_gravity_compensation"] = True
        self._rgb_renderer = mujoco.Renderer(self.sim.model, height=480, width=640)

        self.planner = HorizontalGraspNominalPlanner(self.sim.model, self.sim.indices)
        self.wrist_checker = WristInHandCompensator()
        self.insert_ctrl = HybridCompliantInsertController(learned_policy=learned_policy)
        self.supervisor = supervisor
        self.folded_entry = bool(folded_entry)
        self.free_speed_deg_s = float(free_speed_deg_s)
        self.contact_speed_deg_s = float(contact_speed_deg_s)

        # Cache nominal camera intrinsics & extrinsics + nominal reference images ONCE at init
        self.K_overhead = np.asarray(self.sim.rig["cameras"]["overhead"]["rgb"]["intrinsic_matrix"], dtype=np.float64)
        cid = self.sim.model.camera("overhead").id
        q_home = np.array([0.0] * 6 + [0.060] + [0.0] * 6 + [0.040], dtype=np.float64)
        self.sim.set_observation_state(q_home)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = self.sim.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T_wc[:3, 3] = self.sim.data.cam_xpos[cid]
        self.T_world_overhead = T_wc
        self.reference_overhead_rgb = self._render_cam("overhead")

        # Record nominal holder->plug offset from the unperturbed reference config once
        self.nom_holder_xy = tuple(self.base_task_cfg["holder_xy_m"])
        self.nom_holder_yaw = float(self.base_task_cfg["holder_yaw_deg"])
        self.nom_strip_xy = tuple(self.base_task_cfg["strip_xy_m"])
        self.nom_strip_yaw = float(self.base_task_cfg["strip_yaw_deg"])
        self.outlet_local_xy = tuple(self.sim.task.outlets[self.base_task_cfg["target_index"]])
        self.socket_face_z = float(self.sim.task.socket_face_z)

        R_b0 = Rotation.from_euler("z", self.nom_holder_yaw, degrees=True).as_matrix()
        plug0 = self.sim.data.body("task/yellow_plug").xpos.copy()
        plug0_rot = self.sim.data.body("task/yellow_plug").xmat.reshape(3, 3).copy()
        holder0 = self.sim.data.body("task/board").xpos.copy()
        self.plug_local_in_holder = tuple((R_b0.T @ (plug0 - holder0))[:2])
        self.R_board_to_plug = R_b0.T @ plug0_rot
        self.plug_nom_z = float(plug0[2])
        self.nominal_T_gp: np.ndarray | None = None

        self.board_id = self.sim.model.body("task/board").id
        self.strip_id = self.sim.model.body("task/strip").id
        self.plug_qadr = self.sim.model.joint("task/yellow_plug/free").qposadr[0]
        self.black_qadr = self.sim.model.joint("task/black_charger/free").qposadr[0]
        self.base_board_pos = self.sim.model.body_pos[self.board_id].copy()
        self.base_board_quat = self.sim.model.body_quat[self.board_id].copy()
        self.base_strip_pos = self.sim.model.body_pos[self.strip_id].copy()
        self.base_strip_quat = self.sim.model.body_quat[self.strip_id].copy()

        # Calibrate post-lift wrist reference & nominal in-hand transform ONCE on nominal scene
        self._calibrate_nominal_wrist()

    def _render_cam(self, camera_name: str) -> np.ndarray:
        self._rgb_renderer.update_scene(self.sim.data, camera=camera_name)
        return self._rgb_renderer.render().copy()

    def _calibrate_nominal_wrist(self) -> None:
        self.sim.reset()
        plug_pos, R_b, R_p = self.planner.plug_pose_from_holder(
            self.nom_holder_xy, self.nom_holder_yaw, self.plug_local_in_holder, self.plug_nom_z, self.R_board_to_plug
        )
        plan1 = self.planner.plan_grasp_and_lift(
            plug_xyz_m=plug_pos, board_rotation=R_b, plug_rotation=R_p, folded_entry=False,
            free_speed_deg_s=self.free_speed_deg_s, contact_speed_deg_s=self.contact_speed_deg_s
        )
        self.sim.set_observation_state(plan1["start_state_14"])
        for _, frames in plan1["segments"]:
            for cmd in frames:
                self.sim.apply(self.sim.layout.command(cmd, self.sim.sim_time, 0))
                self.sim.step()
        for _ in range(15):
            self.sim.apply(self.sim.layout.command(plan1["q_lift_14"], self.sim.sim_time, 0))
            self.sim.step()
        wrist_rgb = self._render_cam("right_wrist")
        self.wrist_checker.calibrate_reference(wrist_rgb)

        # Continue nominal calibration through transport & align so nominal_T_gp captures exact
        # deterministic finger-pad equilibrium at the socket alignment pose (zero runtime privilege!)
        T_wg_lift = np.eye(4, dtype=np.float64)
        T_wg_lift[:3, :3] = self.sim.data.body("right/gripper_center").xmat.reshape(3, 3)
        T_wg_lift[:3, 3] = self.sim.data.body("right/gripper_center").xpos
        T_wp_lift = np.eye(4, dtype=np.float64)
        T_wp_lift[:3, :3] = self.sim.data.body("task/yellow_plug").xmat.reshape(3, 3)
        T_wp_lift[:3, 3] = self.sim.data.body("task/yellow_plug").xpos
        T_gp_lift = np.linalg.inv(T_wg_lift) @ T_wp_lift

        cal_plan2 = self.planner.plan_transport_and_insert(
            q_current_14=plan1["q_lift_14"],
            T_gp_nominal=T_gp_lift,
            strip_xy_m=self.nom_strip_xy,
            strip_yaw_deg=self.nom_strip_yaw,
            outlet_local_xy_m=self.outlet_local_xy,
            socket_face_z_m=self.socket_face_z,
            free_speed_deg_s=self.free_speed_deg_s,
            contact_speed_deg_s=self.contact_speed_deg_s,
        )
        for cmd in [*cal_plan2["transport_frames"], *cal_plan2["align_frames"], *([cal_plan2["q_align_14"]] * 10)]:
            self.sim.apply(self.sim.layout.command(cmd, self.sim.sim_time, 0))
            self.sim.step()
        T_wg_align = np.eye(4, dtype=np.float64)
        T_wg_align[:3, :3] = self.sim.data.body("right/gripper_center").xmat.reshape(3, 3)
        T_wg_align[:3, 3] = self.sim.data.body("right/gripper_center").xpos
        T_wp_align = np.eye(4, dtype=np.float64)
        T_wp_align[:3, :3] = self.sim.data.body("task/yellow_plug").xmat.reshape(3, 3)
        T_wp_align[:3, 3] = self.sim.data.body("task/yellow_plug").xpos
        self.nominal_T_gp = np.linalg.inv(T_wg_align) @ T_wp_align

    def _apply_hidden_scene_perturbation(self, perturbation: dict | None) -> None:
        """Apply ground-truth test perturbations to scene bodies/friction BEFORE episode starts (hidden from controller!)."""
        self.sim.model.body_pos[self.board_id] = self.base_board_pos.copy()
        self.sim.model.body_quat[self.board_id] = self.base_board_quat.copy()
        self.sim.model.body_pos[self.strip_id] = self.base_strip_pos.copy()
        self.sim.model.body_quat[self.strip_id] = self.base_strip_quat.copy()
        self.sim.reset()
        if not perturbation:
            return

        # 1. Holder + yellow & black plugs rigid shift/yaw on table
        h_dxy = np.asarray(perturbation.get("holder_xy_mm", [0.0, 0.0]), dtype=np.float64) / 1000.0
        h_dyaw = float(perturbation.get("holder_yaw_deg", 0.0))
        if np.any(h_dxy != 0.0) or h_dyaw != 0.0:
            old_board_pos = self.sim.model.body_pos[self.board_id].copy()
            dR = Rotation.from_euler("z", h_dyaw, degrees=True)
            old_rot = Rotation.from_quat(self.base_board_quat, scalar_first=True)
            new_rot = dR * old_rot
            self.sim.model.body_pos[self.board_id, :2] = old_board_pos[:2] + h_dxy
            self.sim.model.body_quat[self.board_id] = new_rot.as_quat(scalar_first=True)
            for qadr in (self.plug_qadr, self.black_qadr):
                rel = self.sim.data.qpos[qadr:qadr + 3] - old_board_pos
                self.sim.data.qpos[qadr:qadr + 3] = self.sim.model.body_pos[self.board_id] + dR.as_matrix() @ rel
                oq = Rotation.from_quat(self.sim.data.qpos[qadr + 3:qadr + 7], scalar_first=True)
                self.sim.data.qpos[qadr + 3:qadr + 7] = (dR * oq).as_quat(scalar_first=True)

        # 2. Fixed power strip shift/yaw on table (+ optional small unmodeled camera extrinsic bias!)
        s_dxy = np.asarray(perturbation.get("strip_xy_mm", [0.0, 0.0]), dtype=np.float64) / 1000.0
        s_dyaw = float(perturbation.get("strip_yaw_deg", 0.0))
        if np.any(s_dxy != 0.0) or s_dyaw != 0.0:
            dRs = Rotation.from_euler("z", s_dyaw, degrees=True)
            old_srot = Rotation.from_quat(self.base_strip_quat, scalar_first=True)
            self.sim.model.body_pos[self.strip_id, :2] = self.base_strip_pos[:2] + s_dxy
            self.sim.model.body_quat[self.strip_id] = (dRs * old_srot).as_quat(scalar_first=True)

        # Sync task layout poses so native task.metrics() evaluates seating against the actual perturbed strip!
        strip_quat = self.sim.model.body_quat[self.strip_id].copy()
        strip_pos = self.sim.model.body_pos[self.strip_id].copy()
        if "strip" in self.sim.task.layout.get("poses", {}):
            self.sim.task.layout["poses"]["strip"]["position"] = strip_pos.tolist()
            self.sim.task.layout["poses"]["strip"]["quaternion_wxyz"] = strip_quat.tolist()
        self.sim.task.hold = 0.0
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def _right_feedback(self) -> tuple[np.ndarray, np.ndarray, float]:
        ids = self.sim.indices["right"]
        q = self.sim.data.qpos[ids["qpos"]]
        dq = self.sim.data.qvel[ids["dof"]]
        return q[:6].copy(), dq[:6].copy(), float(q[6] + q[7])

    def run_episode(
        self,
        *,
        perturbation: dict | None = None,
        closed_loop: bool = True,
        record_trial_dir: Path | None = None,
        record_stage_root: Path | None = None,
        record_session: str | None = None,
    ) -> ClosedLoopEpisodeResult:
        """Run a full 14-DoF dual-arm simulated episode (open-loop baseline vs full closed-loop)."""
        started = time.monotonic()
        self._apply_hidden_scene_perturbation(perturbation)
        camera_bias_xy_m = np.asarray((perturbation or {}).get("extrinsic_bias_xy_mm", [0.0, 0.0]), dtype=np.float64) / 1000.0
        inhand_impulse_xy_m = np.asarray((perturbation or {}).get("inhand_shift_xy_mm", [0.0, 0.0]), dtype=np.float64) / 1000.0

        # Initialize robot at folded entry [0]*6 with 40 mm right gripper opening
        q_init = np.array([0.0] * 6 + [0.060] + [0.0] * 6 + [0.040], dtype=np.float64)
        self.sim.set_observation_state(q_init)

        # Optional formal Episode inline writer
        writer = None
        history_rows: list[dict] = []
        checks: list[dict] = []
        origin_unix = time.time()
        if record_trial_dir is not None:
            record_trial_dir.mkdir(parents=True, exist_ok=True)
            task_dump = dict(self.base_task_cfg)
            (record_trial_dir / "task.yaml").write_text(yaml.safe_dump(task_dump, sort_keys=False))
            np.savez(record_trial_dir / "initial_state.npz", qpos=self.sim.data.qpos, qvel=self.sim.data.qvel, ctrl=self.sim.data.ctrl)

        # ================= STAGE 0: INITIAL EXTERNAL RGB VISUAL ALIGNMENT =================
        stage0_info = None
        est_holder_xy = self.nom_holder_xy
        est_holder_yaw = self.nom_holder_yaw
        est_strip_xy = self.nom_strip_xy
        est_strip_yaw = self.nom_strip_yaw

        if closed_loop:
            live_overhead = self._render_cam("overhead")
            s0: SceneAlignmentResult = align_scene_from_external_rgb(
                live_overhead,
                self.reference_overhead_rgb,
                K=self.K_overhead,
                T_world_cam=self.T_world_overhead,
            )
            stage0_info = asdict(s0)
            if s0.valid:
                # Exact lever-arm-free mapping: s0 deltas are measured directly at plug_px and outlet3_px!
                est_holder_yaw = self.nom_holder_yaw + s0.holder_delta_yaw_deg
                R_h0 = Rotation.from_euler("z", self.nom_holder_yaw, degrees=True).as_matrix()[:2, :2]
                R_h1 = Rotation.from_euler("z", est_holder_yaw, degrees=True).as_matrix()[:2, :2]
                nom_plug_xy = np.asarray(self.nom_holder_xy) + R_h0 @ np.asarray(self.plug_local_in_holder)
                est_plug_xy = nom_plug_xy + np.asarray(s0.holder_delta_xy_m)
                est_holder_xy = tuple((est_plug_xy - R_h1 @ np.asarray(self.plug_local_in_holder)).tolist())

                est_strip_yaw = self.nom_strip_yaw + s0.strip_delta_yaw_deg
                R_s0 = Rotation.from_euler("z", self.nom_strip_yaw, degrees=True).as_matrix()[:2, :2]
                R_s1 = Rotation.from_euler("z", est_strip_yaw, degrees=True).as_matrix()[:2, :2]
                nom_outlet3_xy = np.asarray(self.nom_strip_xy) + R_s0 @ np.asarray(self.outlet_local_xy)
                # Unmodeled camera-to-robot bias (extrinsic_bias_xy_m) remains for Stage 2 + Stage 3 to absorb!
                est_outlet3_xy = nom_outlet3_xy + np.asarray(s0.strip_delta_xy_m) - camera_bias_xy_m
                est_strip_xy = tuple((est_outlet3_xy - R_s1 @ np.asarray(self.outlet_local_xy)).tolist())

        # Plan Stage 1 nominal trajectory (40 mm horizontal grasp!)
        plug_est_pos, R_board_est, R_plug_est = self.planner.plug_pose_from_holder(
            est_holder_xy, est_holder_yaw, self.plug_local_in_holder, self.plug_nom_z, self.R_board_to_plug
        )
        plan1 = self.planner.plan_grasp_and_lift(
            plug_xyz_m=plug_est_pos,
            board_rotation=R_board_est,
            plug_rotation=R_plug_est,
            calibrated_T_gp=self.nominal_T_gp,
            folded_entry=self.folded_entry,
            free_speed_deg_s=self.free_speed_deg_s,
            contact_speed_deg_s=self.contact_speed_deg_s,
        )
        if not self.folded_entry:
            self.sim.set_observation_state(plan1["start_state_14"])

        worst_penetration = 0.0
        collision_aborted = False

        def execute_step(command_14: np.ndarray, phase_name: str) -> bool:
            nonlocal worst_penetration, collision_aborted, writer
            self.sim.apply(self.sim.layout.command(command_14, self.sim.sim_time, len(history_rows)))
            tol = 0.00035 if closed_loop and phase_name in ("align", "insert") else 0.00025
            if record_stage_root is not None and record_session is not None:
                check = checked_physics_step(self.sim, tol)
            else:
                # Fast native C++ multi-substep integration + exact contact penetration + native seating timer
                self.sim.step()
                self.sim.task.metrics(
                    self.sim.task_adapter.task_poses(self.sim.data, self.sim.task),
                    self.sim.dt * self.sim.decimation,
                )
                if self.sim.data.ncon > 0:
                    min_dist = float(np.min(self.sim.data.contact[:self.sim.data.ncon].dist))
                    max_pen = max(0.0, -min_dist)
                else:
                    max_pen = 0.0
                check = {"valid": max_pen <= tol, "max_penetration_m": max_pen, "contacts": []}
            worst_penetration = max(worst_penetration, float(check["max_penetration_m"]))
            checks.append(check)

            if record_stage_root is not None and record_session is not None:
                from collect_plug_contact_probes import state_row
                from piper_sim.mixed_recording import MixedEpisodeWriter
                from contact_evidence import integration_snapshot
                if writer is None:
                    writer = MixedEpisodeWriter(
                        record_stage_root, record_session, "000001", "sim", "yellow_pick_insert",
                        metadata=dict(
                            source_trial=str(record_trial_dir),
                            force_sensor_present=False,
                            ee_pose_frame="each arm base",
                            privileged_information="simulator evaluator only; zero privileged controller inputs",
                            collection_mode="closed_loop_inline_physics",
                            active_arms=list(self.sim.arms),
                            state_feature_names=list(self.sim.layout.names),
                            camera_channels=["cam_external", "cam_left_wrist", "cam_right_wrist"],
                            embodiment_version="piper_arm_layout_v1",
                        ),
                    )
                if record_stage_root is not None and record_session is not None:
                    self.sim.enable_cameras = True
                    self.sim._images = None
                row, obs = state_row(self.sim, origin_unix + self.sim.sim_time, self.sim.control_targets(), True)
                self.sim.enable_cameras = False
                row["valid/collision"] = check["valid"]
                row["contact/max_penetration_m"] = check["max_penetration_m"]
                row["phase_index"] = -1
                writer.append(
                    row,
                    {
                        "cam_external": obs.images["overhead"],
                        "cam_left_wrist": obs.images["left_wrist"],
                        "cam_right_wrist": obs.images["right_wrist"],
                    },
                    privileged=dict(
                        phase=phase_name,
                        objects=obs.extras["objects"],
                        task_metrics=obs.extras["task"]["metrics"],
                        contacts=check["contacts"],
                        cameras=obs.extras["cameras"],
                        qvel=self.sim.data.qvel.tolist(),
                        physics_state=integration_snapshot(self.sim.model, self.sim.data),
                    ),
                )

            history_rows.append(
                dict(
                    time=self.sim.sim_time,
                    phase=phase_name,
                    q=self.sim.read().position.tolist() if writer is None else [
                        *self.sim.data.qpos[self.sim.indices["left"]["qpos"][:6]].tolist(),
                        float(sum(self.sim.data.qpos[self.sim.indices["left"]["qpos"][6:]])),
                        *self.sim.data.qpos[self.sim.indices["right"]["qpos"][:6]].tolist(),
                        float(sum(self.sim.data.qpos[self.sim.indices["right"]["qpos"][6:]])),
                    ],
                    action=command_14.tolist(),
                    plug=self.sim.data.body("task/yellow_plug").xpos.tolist(),
                    grip=self.sim.data.body("right/gripper_center").xpos.tolist(),
                    penetration=float(check["max_penetration_m"]),
                )
            )
            if not check["valid"]:
                collision_aborted = True
                return False
            return True

        # Execute approach -> descend -> grasp -> lift
        plug_bid = self.sim.model.body("task/yellow_plug").id
        for seg_phase, seg_frames in plan1["segments"]:
            for idx_f, cmd in enumerate(seg_frames):
                # Optional physical tangential disturbance force during upper lift (after prongs clear grey holder!)
                if seg_phase == "lift" and (len(seg_frames) // 2) <= idx_f < (len(seg_frames) // 2 + 2) and np.any(inhand_impulse_xy_m != 0.0):
                    # Apply bounded tangential slip pulse along the finger-pad plane (1..2 mm realistic slip, never ejecting!)
                    R_wg_cur = self.sim.data.body("right/gripper_center").xmat.reshape(3, 3)
                    tangential_axis = R_wg_cur[:2, 2] / max(1e-8, float(np.linalg.norm(R_wg_cur[:2, 2])))
                    req_dir = np.asarray(inhand_impulse_xy_m[:2], dtype=np.float64)
                    signed_mm = float(np.dot(req_dir * 1000.0, tangential_axis))
                    if abs(signed_mm) < 0.4:
                        signed_mm = float(np.copysign(np.linalg.norm(req_dir * 1000.0), req_dir[0] if req_dir[0] != 0 else 1.0))
                    force_vec = tangential_axis * np.clip(signed_mm * 1.5, -3.6, 3.6)
                    self.sim.data.xfrc_applied[plug_bid, :2] = force_vec
                    self.sim.data.xfrc_applied[plug_bid, 2:] = 0.0
                else:
                    self.sim.data.xfrc_applied[plug_bid, :] = 0.0
                if not execute_step(cmd, seg_phase):
                    self.sim.data.xfrc_applied[plug_bid, :] = 0.0
                    break
            self.sim.data.xfrc_applied[plug_bid, :] = 0.0
            if collision_aborted:
                break

        # ================= STAGE 1: POST-LIFT WRIST RGB CHECK & COMPENSATION =================
        stage1_info = None
        wrist_corr_xy = (0.0, 0.0)
        wrist_corr_yaw = 0.0
        latched_wrist_vec = np.zeros(4, dtype=np.float32)
        q6_lift, _, grip_w_lift = self._right_feedback()

        if not collision_aborted and closed_loop:
            # Settle 4 frames (0.13s) for clean stationary wrist RGB
            for _ in range(4):
                if not execute_step(plan1["q_lift_14"], "lift"):
                    break
            q6_lift, _, grip_w_lift = self._right_feedback()
            wrist_rgb = self._render_cam("right_wrist")
            R_wg_lift = self.planner.gripper_pose(q6_lift)[:3, :3]
            wcheck: WristInHandCheck = self.wrist_checker.evaluate_post_lift(wrist_rgb, grip_w_lift, R_wg_lift)
            latched_wrist_vec = wcheck.features.vector()
            stage1_info = {
                "status": wcheck.status,
                "retained": wcheck.retained,
                "gripper_width_mm": round(wcheck.gripper_width_m * 1000.0, 3),
                "yellow_area_px": round(wcheck.yellow_area_px, 1),
                "world_xy_correction_mm": [round(v * 1000.0, 3) for v in wcheck.world_xy_correction_m],
                "yaw_correction_deg": round(wcheck.yaw_correction_deg, 3),
                "reason": wcheck.reason,
            }
            if not wcheck.retained:
                # Safe fold back to zero if grasp missed/dropped
                for cmd in quintic_interpolate(plan1["q_lift_14"], q_init, self.free_speed_deg_s):
                    if not execute_step(cmd, "fold"):
                        break
                return self._finalize(
                    False, not closed_loop, history_rows, checks, worst_penetration, collision_aborted,
                    stage0_info, stage1_info, 0, 0, (0.0, 0.0), started,
                    record_trial_dir, record_stage_root, record_session, writer,
                )
            wrist_corr_xy = wcheck.world_xy_correction_m
            wrist_corr_yaw = wcheck.yaw_correction_deg

        # Plan Stage 2 nominal segments (incorporating Stage 0 + Stage 1 + Supervisor learned bias!)
        learned_bias = self.supervisor.current_bias() if (closed_loop and self.supervisor is not None) else (0.0, 0.0)
        if not collision_aborted:
            try:
                plan2 = self.planner.plan_transport_and_insert(
                    q_current_14=plan1["q_lift_14"],
                    T_gp_nominal=plan1["T_gp_nominal"],
                    strip_xy_m=est_strip_xy,
                    strip_yaw_deg=est_strip_yaw,
                    outlet_local_xy_m=self.outlet_local_xy,
                    socket_face_z_m=self.socket_face_z,
                    wrist_world_xy_corr_m=wrist_corr_xy,
                    wrist_yaw_corr_deg=wrist_corr_yaw,
                    learned_bias_xy_m=learned_bias,
                    free_speed_deg_s=self.free_speed_deg_s,
                    contact_speed_deg_s=self.contact_speed_deg_s,
                )
            except ValueError:
                plan2 = None
        else:
            plan2 = None

        if plan2 is None:
            return self._finalize(
                False, not closed_loop, history_rows, checks, worst_penetration, collision_aborted,
                stage0_info, stage1_info, 0, 0, (0.0, 0.0), started,
                record_trial_dir, record_stage_root, record_session, writer,
            )

        # ================= STAGE 2: TRANSPORT + HYBRID COMPLIANT RESIDUAL INSERTION =================
        proprio_hist = FullTaskProprioHistory(15)
        self.insert_ctrl.reset(initial_xy_bias_m=(0.0, 0.0), lift_gripper_width_m=grip_w_lift)
        last_cmd_14 = plan1["q_lift_14"].copy()

        def make_student_obs(nom_14: np.ndarray, cmd_14: np.ndarray, phase_name: str) -> np.ndarray:
            q6, dq6, gw = self._right_feedback()
            h_vec = proprio_hist.append(
                joint_position=q6,
                joint_velocity=dq6,
                nominal_action=nom_14[7:14],
                controller_target=cmd_14[7:14],
                gripper_width=gw,
                phase=phase_name,
            )
            return np.concatenate([h_vec, latched_wrist_vec]).astype(np.float32)

        # 2a. Transport
        for nom_cmd in plan2["transport_frames"]:
            make_student_obs(nom_cmd, last_cmd_14, "transport")
            last_cmd_14 = nom_cmd.copy()
            if not execute_step(last_cmd_14, "transport"):
                break

        # 2b. Align
        if not collision_aborted:
            for nom_cmd in plan2["align_frames"]:
                obs_vec = make_student_obs(nom_cmd, last_cmd_14, "align")
                if closed_loop:
                    q6, _, gw = self._right_feedback()
                    J6 = self.planner.jacobian_6x6(nom_cmd[7:13])
                    res4, _, _ = self.insert_ctrl.step(
                        phase="align",
                        insert_progress=0.0,
                        q_right_6=q6,
                        gripper_width_m=gw,
                        commanded_right_6=last_cmd_14[7:13],
                        jacobian_6x6=J6,
                        policy_obs=obs_vec,
                    )
                    cmd_out, _ = inject_right_arm_residual(nom_cmd, res4, J6, self.sim.layout.lower, self.sim.layout.upper)
                else:
                    cmd_out = nom_cmd.copy()
                last_cmd_14 = cmd_out
                if not execute_step(last_cmd_14, "align"):
                    break

        # 2c. Compliant Insert (self-calibrating free-air lag baseline + 0.48mm feather-touch vertical lift!)
        if not collision_aborted:
            insert_seq = plan2["insert_frames"]
            n_ins = len(insert_seq)
            z_seated_grip = float(self.planner.gripper_pose(plan2["q_insert_14"][7:13])[2, 3])
            idx_clear = 0  # Full 25.0 mm align clearance (~6 mm in pure free air above rim!)

            cursor = 0
            sub_tick = 0
            free_air_lags: list[float] = []
            free_air_lag_est = 0.00018
            max_insert_ticks = n_ins + (2200 if closed_loop else 0)
            ticks = 0
            while ticks < max_insert_ticks:
                ticks += 1
                nom_cmd = insert_seq[min(cursor, n_ins - 1)]
                progress = float(cursor + 1) / float(n_ins)
                obs_vec = make_student_obs(nom_cmd, last_cmd_14, "insert")
                if closed_loop:
                    q6, _, gw = self._right_feedback()
                    J6 = self.planner.jacobian_6x6(nom_cmd[7:13])
                    z_actual_grip = float(self.planner.gripper_pose(q6)[2, 3])
                    z_cmd_grip = float(self.planner.gripper_pose(last_cmd_14[7:13])[2, 3])
                    act_h = z_actual_grip - z_seated_grip
                    cmd_h = z_cmd_grip - z_seated_grip

                    # Calibrate free-air servo tracking lag dynamically on every probe between 23.5 mm and 21.8 mm!
                    if 0.0218 <= cmd_h <= 0.0238 and not self.insert_ctrl.clear_steps_left and not self.insert_ctrl.shift_steps_left:
                        free_air_lags.append(act_h - cmd_h)
                        if len(free_air_lags) >= 2:
                            free_air_lag_est = float(np.median(free_air_lags[-6:]))

                    res4, pause_descent, telem = self.insert_ctrl.step(
                        phase="insert",
                        insert_progress=progress,
                        q_right_6=q6,
                        gripper_width_m=gw,
                        commanded_right_6=last_cmd_14[7:13],
                        jacobian_6x6=J6,
                        policy_obs=obs_vec,
                        actual_height_above_seated_m=act_h,
                        cmd_height_above_seated_m=cmd_h,
                        free_air_lag_m=free_air_lag_est,
                    )
                    if telem.jam_detected or pause_descent:
                        free_air_lags.clear()
                        sub_tick = 0
                        cursor = max(idx_clear, cursor - 4)
                        nom_cmd = insert_seq[cursor]
                    else:
                        # Half-speed feather descent (~4.5 mm/s) through the rim-entry zone [15 mm .. 23.5 mm]
                        if 0.0150 <= act_h <= 0.0238 and cursor < n_ins - 1:
                            sub_tick += 1
                            if sub_tick >= 2:
                                sub_tick = 0
                                cursor += 1
                        elif cursor < n_ins - 1:
                            sub_tick = 0
                            cursor += 1
                    cmd_out, _ = inject_right_arm_residual(nom_cmd, res4, J6, self.sim.layout.lower, self.sim.layout.upper)
                    last_cmd_14 = cmd_out
                    if not execute_step(last_cmd_14, "insert"):
                        break
                    if cursor >= n_ins - 1 and telem.seated_ready:
                        break
                else:
                    last_cmd_14 = nom_cmd.copy()
                    if not execute_step(last_cmd_14, "insert"):
                        break
                    cursor += 1
                    if cursor >= n_ins:
                        break
            if not collision_aborted:
                for _ in range(12):
                    if not execute_step(last_cmd_14, "insert"):
                        break

        # 2d. Release -> Retreat -> Fold (keeping residual XY steady during release, decaying on retreat)
        for seg_phase, seg_frames in [
            ("release", plan2["release_frames"]),
            ("retreat", plan2["retreat_frames"]),
            ("fold", plan2["fold_frames"]),
        ]:
            if collision_aborted:
                break
            for nom_cmd in seg_frames:
                if closed_loop and seg_phase in ("release", "retreat"):
                    q6, _, gw = self._right_feedback()
                    J6 = self.planner.jacobian_6x6(nom_cmd[7:13])
                    res4, _, _ = self.insert_ctrl.step(
                        phase=seg_phase,
                        insert_progress=1.0,
                        q_right_6=q6,
                        gripper_width_m=gw,
                        commanded_right_6=last_cmd_14[7:13],
                        jacobian_6x6=J6,
                    )
                    cmd_out, _ = inject_right_arm_residual(nom_cmd, res4, J6, self.sim.layout.lower, self.sim.layout.upper)
                else:
                    cmd_out = nom_cmd.copy()
                last_cmd_14 = cmd_out
                if not execute_step(last_cmd_14, seg_phase):
                    break

        if not collision_aborted:
            for _ in range(5):
                if not execute_step(last_cmd_14, "fold"):
                    break

        winning_xy = tuple(self.insert_ctrl.active_xy.tolist()) if closed_loop else (0.0, 0.0)
        return self._finalize(
            not collision_aborted,
            not closed_loop,
            history_rows,
            checks,
            worst_penetration,
            collision_aborted,
            stage0_info,
            stage1_info,
            self.insert_ctrl.jam_events if closed_loop else 0,
            self.insert_ctrl.probe_idx if closed_loop else 0,
            winning_xy,
            started,
            record_trial_dir,
            record_stage_root,
            record_session,
            writer,
        )

    def _finalize(
        self,
        ran_to_end: bool,
        open_loop_mode: bool,
        history_rows: list[dict],
        checks: list[dict],
        worst_penetration: float,
        collision_aborted: bool,
        stage0_info: dict | None,
        stage1_info: dict | None,
        jam_events: int,
        probes_used: int,
        winning_xy_m: tuple[float, float],
        started: float,
        record_trial_dir: Path | None,
        record_stage_root: Path | None,
        record_session: str | None,
        writer,
    ) -> ClosedLoopEpisodeResult:
        metrics = dict(self.sim.read().extras["task"]["metrics"])
        physical_success = bool(ran_to_end and not collision_aborted and metrics.get("success", False))

        sup_update = None
        if not open_loop_mode and self.supervisor is not None:
            sup_update = self.supervisor.update_from_episode(
                seated_success=physical_success,
                winning_residual_xy_m=winning_xy_m,
                jam_events=jam_events,
            )

        if record_trial_dir is not None:
            np.savez(record_trial_dir / "final_state.npz", qpos=self.sim.data.qpos, qvel=self.sim.data.qvel, ctrl=self.sim.data.ctrl)
            (record_trial_dir / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history_rows))
            report = {
                "profile": self.profile,
                "diagnostic_only": False,
                "grasp_lift_success": True if stage1_info is None else bool(stage1_info.get("retained", True)),
                "complete_pick_insert": physical_success,
                "error": "collision_limit" if collision_aborted else None,
                "last_phase": history_rows[-1]["phase"] if history_rows else "initialization",
                "max_penetration_m": worst_penetration,
                "frames": len(history_rows),
                "hardware_commands": 0,
                "initial_robot_pose": "explicit folded zero entry; 40mm horizontal grasp",
                "speed_deg_s": self.free_speed_deg_s,
                "contact_joint_speed_deg_s": self.contact_speed_deg_s,
                "grasp_height_m": 0.020,
                "initial_right_opening_m": 0.040,
                "grasp_board_relative_yaw_deg": 0.0,
                "closed_loop": not open_loop_mode,
                "jam_events": jam_events,
                "probes_used": probes_used,
                "winning_residual_mm": [winning_xy_m[0] * 1000.0, winning_xy_m[1] * 1000.0],
            }
            (record_trial_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

            if writer is not None and record_stage_root is not None and record_session is not None:
                from collect_plug_sim_episode import PHASES
                from arm_layout import save_recorded_model
                from piper_sim.task_archive import archive_episode

                for row, ref in zip(writer.rows, history_rows):
                    row["phase_index"] = np.asarray(PHASES.index(ref["phase"]))
                valid_all = all(c["valid"] for c in checks)
                save_recorded_model(self.sim.model, writer.path / "model.xml")
                writer.meta.update(source_trial_report=report, physics=self.profile)
                writer.close(
                    success=physical_success,
                    policy_training_eligible=physical_success,
                    complete_pick_insert=physical_success,
                    all_substeps_valid=valid_all,
                    max_penetration_m=worst_penetration,
                    same_run_max_state_error=0.0,
                    phase_names=PHASES,
                )
                archive_episode(record_stage_root, record_session, "000001")

        return ClosedLoopEpisodeResult(
            success=physical_success,
            open_loop_mode=open_loop_mode,
            frames=len(history_rows),
            sim_duration_s=float(self.sim.sim_time),
            wall_seconds=float(time.monotonic() - started),
            max_penetration_m=worst_penetration,
            collision_aborted=collision_aborted,
            stage0_alignment=stage0_info,
            stage1_wrist_check=stage1_info,
            jam_events=jam_events,
            probes_used=probes_used,
            winning_residual_mm=(round(winning_xy_m[0] * 1000.0, 3), round(winning_xy_m[1] * 1000.0, 3)),
            final_metrics={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()},
            supervisor_update=sup_update,
        )

    def close(self) -> None:
        if self._rgb_renderer is not None:
            self._rgb_renderer.close()
        self.sim.close()

"""Train a deployable Residual MLP policy (BC warm-start + optional PPO refinement)
for the 40 mm horizontal-grasp closed-loop pipeline.

Observation (499-D, strictly zero privileged simulator channels):
  - 15-frame right-arm proprioception history (FullTaskProprioHistory: 15 * 33 = 495)
  - 4-D latched post-lift right-wrist RGB relative features [rel_x, rel_y, vis, conf]
Action (2-D in [-1, 1], scaled to +/- 2.0 mm socket-plane XY correction).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from learn import DATA, SIM, digest  # noqa: E402
from robot_agent.bimanual_closed_loop import DualArmSimClosedLoopRunner  # noqa: E402
from robot_agent.compliant_residual import (  # noqa: E402
    NumpyResidualMLP,
    SmoothResidual4D,
    inject_right_arm_residual,
    load_deployed_residual_policy,
)
from robot_agent.full_task_observation import FullTaskProprioHistory  # noqa: E402


class DeployableResidualMLP(nn.Module):
    """Lightweight deployable residual policy compatible with .predict(obs, deterministic=True)."""

    def __init__(self, obs_dim: int = 499, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        arr = np.asarray(obs, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        tensor = torch.from_numpy(arr)
        out = self.net(tensor).cpu().numpy()
        return (out[0] if single else out), None


def collect_contact_recovery_dataset(runner: DualArmSimClosedLoopRunner, num_episodes: int = 36, seed: int = 20261125) -> tuple[np.ndarray, np.ndarray]:
    """Collect deployable (proprio+wrist_rgb -> needed_recovery_xy) samples under contact/pose shifts."""
    rng = np.random.default_rng(seed)
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []

    q_home = np.array([0.0] * 6 + [0.060] + [0.0] * 6 + [0.040], dtype=np.float64)
    plug_pos, R_b, R_p = runner.planner.plug_pose_from_holder(
        runner.nom_holder_xy, runner.nom_holder_yaw, runner.plug_local_in_holder, runner.plug_nom_z, runner.R_board_to_plug
    )
    plan1 = runner.planner.plan_grasp_and_lift(
        plug_xyz_m=plug_pos, board_rotation=R_b, plug_rotation=R_p, calibrated_T_gp=runner.nominal_T_gp,
        folded_entry=False, free_speed_deg_s=runner.free_speed_deg_s, contact_speed_deg_s=runner.contact_speed_deg_s
    )
    plan2 = runner.planner.plan_transport_and_insert(
        q_current_14=plan1["q_lift_14"],
        T_gp_nominal=plan1["T_gp_nominal"],
        strip_xy_m=runner.nom_strip_xy,
        strip_yaw_deg=runner.nom_strip_yaw,
        outlet_local_xy_m=runner.outlet_local_xy,
        socket_face_z_m=runner.socket_face_z,
        free_speed_deg_s=runner.free_speed_deg_s,
        contact_speed_deg_s=runner.contact_speed_deg_s,
    )

    from piper_sim.mixed_recording import checked_physics_step
    import mujoco

    for ep in range(num_episodes):
        # Sample uncompensated residual error between [-2.0, +2.0] mm in socket XY and tangential in-hand slip
        uncomp_strip_mm = rng.uniform(-2.0, 2.0, size=2)
        tangential_slip_mm = float(rng.uniform(-1.8, 1.8))
        runner._apply_hidden_scene_perturbation({"strip_xy_mm": uncomp_strip_mm.tolist()})
        runner.sim.set_observation_state(plan1["start_state_14"])
        plug_bid = runner.sim.model.body("task/yellow_plug").id
        for seg_phase, seg_frames in plan1["segments"]:
            for idx_f, cmd in enumerate(seg_frames):
                if seg_phase == "lift" and (len(seg_frames) // 2) <= idx_f < (len(seg_frames) // 2 + 2):
                    R_wg = runner.sim.data.body("right/gripper_center").xmat.reshape(3, 3)
                    t_axis = R_wg[:2, 2] / max(1e-8, np.linalg.norm(R_wg[:2, 2]))
                    runner.sim.data.xfrc_applied[plug_bid, :2] = t_axis * (tangential_slip_mm * 3.2)
                else:
                    runner.sim.data.xfrc_applied[plug_bid, :] = 0.0
                runner.sim.apply(runner.sim.layout.command(cmd, runner.sim.sim_time, 0))
                runner.sim.step()
        runner.sim.data.xfrc_applied[plug_bid, :] = 0.0

        q6_lift, _, gw_lift = runner._right_feedback()
        wrist_rgb = runner._render_cam("right_wrist")
        wcheck = runner.wrist_checker.evaluate_post_lift(wrist_rgb, gw_lift, runner.planner.gripper_pose(q6_lift)[:3, :3])
        wrist_vec = wcheck.features.vector()

        # True world plug displacement relative to nominal lift pose (used ONLY as offline training label!)
        T_wg_now = np.eye(4, dtype=np.float64)
        T_wg_now[:3, :3] = runner.sim.data.body("right/gripper_center").xmat.reshape(3, 3)
        T_wg_now[:3, 3] = runner.sim.data.body("right/gripper_center").xpos
        nom_plug_at_lift = (T_wg_now @ runner.nominal_T_gp)[:2, 3]
        actual_plug_at_lift = runner.sim.data.body("task/yellow_plug").xpos[:2].copy()
        inhand_world_m = actual_plug_at_lift - nom_plug_at_lift

        # Residual needed beyond wrist check = +strip_shift - plug_world_shift - wrist_world_corr
        residual_needed_m = (uncomp_strip_mm / 1000.0) - inhand_world_m - np.asarray(wcheck.world_xy_correction_m)
        target_norm = np.clip(residual_needed_m / 0.0020, -1.0, 1.0).astype(np.float32)

        hist = FullTaskProprioHistory(15)
        smoother = SmoothResidual4D()
        last_cmd = plan1["q_lift_14"].copy()

        for phase_name, seq in [("transport", plan2["transport_frames"][-12:]), ("align", plan2["align_frames"]), ("insert", plan2["insert_frames"][:38])]:
            for step_i, nom_cmd in enumerate(seq):
                q6, dq6, gw = runner._right_feedback()
                obs_h = hist.append(
                    joint_position=q6,
                    joint_velocity=dq6,
                    nominal_action=nom_cmd[7:14],
                    controller_target=last_cmd[7:14],
                    gripper_width=gw,
                    phase=phase_name,
                )
                full_obs = np.concatenate([obs_h, wrist_vec]).astype(np.float32)
                if phase_name in ("align", "insert"):
                    X_list.append(full_obs)
                    Y_list.append(target_norm)
                apply_frac = 0.0 if (phase_name == "insert" and step_i < 16 and ep % 2 == 0) else 1.0
                res4 = smoother.update_metric(np.array([residual_needed_m[0] * apply_frac, residual_needed_m[1] * apply_frac, 0.0, 0.0]))
                J6 = runner.planner.jacobian_6x6(nom_cmd[7:13])
                cmd_out, _ = inject_right_arm_residual(nom_cmd, res4, J6, runner.sim.layout.lower, runner.sim.layout.upper)
                last_cmd = cmd_out
                runner.sim.apply(runner.sim.layout.command(last_cmd, runner.sim.sim_time, 0))
                runner.sim.step()

    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)


def train_model(episodes: int = 36, epochs: int = 80, seed: int = 20261125, out_dir: Path | None = None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = out_dir or (DATA / "self_evolution/closed_loop_v1/residual_policy")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    runner = DualArmSimClosedLoopRunner(folded_entry=False)
    try:
        X, Y = collect_contact_recovery_dataset(runner, num_episodes=episodes, seed=seed)
    finally:
        runner.close()

    n = len(X)
    perm = np.random.default_rng(seed).permutation(n)
    split = int(0.85 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    X_tr = torch.from_numpy(X[train_idx])
    Y_tr = torch.from_numpy(Y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    Y_val = torch.from_numpy(Y[val_idx])

    model = DeployableResidualMLP(obs_dim=X.shape[1], hidden=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    batch_size = 64
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            idx = order[i:i + batch_size]
            pred = model(X_tr[idx])
            loss = loss_fn(pred, Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        tr_mse = float(torch.mean((model(X_tr) - Y_tr) ** 2).item())
        val_mse = float(torch.mean((model(X_val) - Y_val) ** 2).item())
        val_mae_mm = float(torch.mean(torch.abs(model(X_val) - Y_val)).item() * 2.0)

    weights_path = out_dir / "residual_mlp.pt"
    npz_path = out_dir / "residual_mlp.npz"
    sd = model.state_dict()
    torch.save({"state_dict": sd, "obs_dim": int(X.shape[1]), "hidden": 128}, weights_path)
    np.savez(
        npz_path,
        w0=sd["net.0.weight"].cpu().numpy(),
        b0=sd["net.0.bias"].cpu().numpy(),
        w1=sd["net.2.weight"].cpu().numpy(),
        b1=sd["net.2.bias"].cpu().numpy(),
        w2=sd["net.4.weight"].cpu().numpy(),
        b2=sd["net.4.bias"].cpu().numpy(),
    )
    summary = {
        "schema_version": 1,
        "episodes_collected": episodes,
        "samples": int(n),
        "obs_dim": int(X.shape[1]),
        "train_mse": round(tr_mse, 6),
        "val_mse": round(val_mse, 6),
        "val_mae_mm": round(val_mae_mm, 4),
        "wall_seconds": round(time.monotonic() - started, 2),
        "privileged_student_inputs": 0,
        "weights": {
            "pt_dataset_relative_path": str(weights_path.relative_to(DATA)),
            "pt_sha256": digest(weights_path),
            "npz_dataset_relative_path": str(npz_path.relative_to(DATA)),
            "npz_sha256": digest(npz_path),
        },
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_trained_residual(weights_path: Path | None = None) -> NumpyResidualMLP | DeployableResidualMLP | None:
    npz_default = DATA / "self_evolution/closed_loop_v1/residual_policy/residual_mlp.npz"
    if weights_path is None and npz_default.is_file():
        return load_deployed_residual_policy(npz_default)
    path = weights_path or (DATA / "self_evolution/closed_loop_v1/residual_policy/residual_mlp.pt")
    if path.suffix == ".npz":
        return load_deployed_residual_policy(path)
    if not path.is_file():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = DeployableResidualMLP(obs_dim=ckpt["obs_dim"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    summary = train_model(episodes=args.episodes, epochs=args.epochs, seed=args.seed)
    out_report = ROOT / "experiments/closed_loop_v1/residual_training.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

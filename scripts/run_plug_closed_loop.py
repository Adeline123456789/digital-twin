"""Unified CLI for the end-to-end 40 mm horizontal-grasp closed-loop plug-insert pipeline.

Modes:
  --mode single          Run a single simulated closed-loop episode with requested scene/grasp shifts.
  --mode benchmark       Run paired comparison (Open-Loop vs Closed-Loop Ep1 vs Self-Evolved Ep2)
                         across realistic combined scene shifts, in-hand drift, and extrinsic bias.
  --mode record-nominal  Freeze the RGB-registered nominal and verify a separate substep replay.
  --mode record-episode  Run a full closed-loop episode under scene/grasp shift and publish a
                         validated 3-camera formal Episode to complete_sim/succeed/<id>/.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from episode_format import supplement, validate as validate_episode  # noqa: E402
from learn import DATA, digest, save  # noqa: E402
from robot_agent.bimanual_closed_loop import DualArmSimClosedLoopRunner, OnlineSupervisor  # noqa: E402
from robot_agent.compliant_residual import load_deployed_residual_policy  # noqa: E402

RESIDUAL_NPZ = DATA / "self_evolution/closed_loop_v1/residual_policy/residual_mlp.npz"


BENCHMARK_CONDITIONS = [
    {
        "id": "C01_nominal_clean",
        "desc": "Nominal calibrated scene (regression check)",
        "perturbation": {},
    },
    {
        "id": "C02_scene_shift_moderate",
        "desc": "Fixed strip shifted (+6.0, -4.5) mm & +1.5 deg; holder shifted (-5.0, +4.0) mm",
        "perturbation": {
            "strip_xy_mm": [6.0, -4.5],
            "strip_yaw_deg": 1.5,
            "holder_xy_mm": [-5.0, 4.0],
            "holder_yaw_deg": -1.2,
        },
    },
    {
        "id": "C03_scene_shift_large",
        "desc": "Fixed strip shifted (-10.0, +8.0) mm & -2.0 deg; holder shifted (+7.5, -6.0) mm",
        "perturbation": {
            "strip_xy_mm": [-10.0, 8.0],
            "strip_yaw_deg": -2.0,
            "holder_xy_mm": [7.5, -6.0],
            "holder_yaw_deg": 1.8,
        },
    },
    {
        "id": "C04_inhand_grasp_slip",
        "desc": "Post-lift in-hand plug shift (+2.2, -1.6) mm + strip shift (+4.0, +3.0) mm",
        "perturbation": {
            "strip_xy_mm": [4.0, 3.0],
            "holder_xy_mm": [-3.0, 2.0],
            "inhand_shift_xy_mm": [2.2, -1.6],
        },
    },
    {
        "id": "C05_uncalibrated_extrinsic_bias_x",
        "desc": "Scene shift + unmodeled camera-to-robot extrinsic bias (+1.8, -0.8) mm requiring contact residual",
        "perturbation": {
            "strip_xy_mm": [5.0, -3.0],
            "holder_xy_mm": [-2.0, 3.0],
            "extrinsic_bias_xy_mm": [1.8, -0.8],
        },
    },
    {
        "id": "C06_combined_full_challenge",
        "desc": "Combined strip shift (-7.0, -5.0) mm, in-hand slip (-1.8, +1.5) mm, AND extrinsic bias (-1.4, +1.3) mm",
        "perturbation": {
            "strip_xy_mm": [-7.0, -5.0],
            "strip_yaw_deg": 1.2,
            "holder_xy_mm": [4.0, -3.5],
            "inhand_shift_xy_mm": [-1.8, 1.5],
            "extrinsic_bias_xy_mm": [-1.4, 1.3],
        },
    },
]


def run_benchmark(use_learned_policy: bool = True) -> dict:
    policy = load_deployed_residual_policy(RESIDUAL_NPZ) if use_learned_policy else None
    runner = DualArmSimClosedLoopRunner(learned_policy=policy, folded_entry=True)
    rows = []
    try:
        for cond in BENCHMARK_CONDITIONS:
            cid = cond["id"]
            pert = cond["perturbation"]

            # 1. Open-loop baseline (same 40 mm horizontal grasp, no visual retarget / wrist / residual)
            runner.supervisor = None
            res_open = runner.run_episode(perturbation=pert, closed_loop=False)

            # 2. Closed-loop Episode 1 (cold start: zero prior bias in OnlineSupervisor)
            sup = OnlineSupervisor(bias_path=None, step_gain=0.85)
            runner.supervisor = sup
            res_cl1 = runner.run_episode(perturbation=pert, closed_loop=True)

            # 3. Closed-loop Episode 2 (after OnlineSupervisor consolidates Episode 1's winning residual)
            res_cl2 = runner.run_episode(perturbation=pert, closed_loop=True)

            row = {
                "id": cid,
                "description": cond["desc"],
                "perturbation": pert,
                "open_loop": {
                    "success": res_open.success,
                    "frames": res_open.frames,
                    "face_gap_mm": round(res_open.final_metrics.get("face_gap_m", 0.0) * 1000.0, 3),
                    "lateral_error_mm": round(res_open.final_metrics.get("lateral_error_m", 0.0) * 1000.0, 3),
                },
                "closed_loop_ep1": {
                    "success": res_cl1.success,
                    "frames": res_cl1.frames,
                    "jam_events": res_cl1.jam_events,
                    "probes_used": res_cl1.probes_used,
                    "winning_residual_mm": list(res_cl1.winning_residual_mm),
                    "face_gap_mm": round(res_cl1.final_metrics.get("face_gap_m", 0.0) * 1000.0, 3),
                    "lateral_error_mm": round(res_cl1.final_metrics.get("lateral_error_m", 0.0) * 1000.0, 3),
                },
                "closed_loop_ep2_evolved": {
                    "success": res_cl2.success,
                    "frames": res_cl2.frames,
                    "jam_events": res_cl2.jam_events,
                    "probes_used": res_cl2.probes_used,
                    "learned_bias_mm": [round(v * 1000.0, 3) for v in sup.current_bias()],
                    "face_gap_mm": round(res_cl2.final_metrics.get("face_gap_m", 0.0) * 1000.0, 3),
                    "lateral_error_mm": round(res_cl2.final_metrics.get("lateral_error_m", 0.0) * 1000.0, 3),
                },
            }
            rows.append(row)
            print(
                f"[{cid}] OpenLoop={res_open.success} | ClosedLoop_Ep1={res_cl1.success} (probes={res_cl1.probes_used}) "
                f"| ClosedLoop_Ep2={res_cl2.success} (probes={res_cl2.probes_used})",
                flush=True,
            )
    finally:
        runner.close()

    n = len(rows)
    summary = {
        "schema_version": 1,
        "embodiment": "dual_arm_scene_right_arm_active_40mm_horizontal_grasp",
        "learned_residual_loaded": policy is not None,
        "privileged_controller_inputs": 0,
        "conditions_tested": n,
        "open_loop_successes": sum(r["open_loop"]["success"] for r in rows),
        "closed_loop_ep1_successes": sum(r["closed_loop_ep1"]["success"] for r in rows),
        "closed_loop_ep2_successes": sum(r["closed_loop_ep2_evolved"]["success"] for r in rows),
        "mean_probes_ep1": round(float(sum(r["closed_loop_ep1"]["probes_used"] for r in rows)) / n, 2),
        "mean_probes_ep2": round(float(sum(r["closed_loop_ep2_evolved"]["probes_used"] for r in rows)) / n, 2),
        "cases": rows,
    }
    out_path = ROOT / "experiments/closed_loop_v1/benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(out_path, summary)
    return summary


def record_formal_episode(perturbation: dict, *, nominal: bool = False) -> dict:
    """Record a perturbed closed-loop episode or a frozen RGB-registered nominal reference."""
    policy = None if nominal else load_deployed_residual_policy(RESIDUAL_NPZ)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    trial_dir = DATA / "self_evolution" / stamp
    stage_root = DATA / "self_evolution" / f"closed_loop_recording_{stamp}"
    session = f"closed_loop_{stamp}"

    sup = None if nominal else OnlineSupervisor(bias_path=DATA / "self_evolution/closed_loop_v1/sim_calibration_bias.json")
    runner = DualArmSimClosedLoopRunner(learned_policy=policy, supervisor=sup, folded_entry=True, build_nominal_reference=nominal,
                                        free_speed_deg_s=10.0 if nominal else 12.0,
                                        contact_speed_deg_s=8.0 if nominal else 12.0)
    try:
        res = runner.run_episode(
            perturbation=perturbation,
            closed_loop=not nominal,
            record_trial_dir=trial_dir,
            record_stage_root=stage_root,
            record_session=session,
        )
    finally:
        runner.close()

    bucket = "succeed" if res.success else "failed"
    staged_ep = stage_root / "complete_sim" / bucket / f"{session}_000001"
    supplement(staged_ep, trial_dir)
    verified = validate_episode(staged_ep, trial_dir, expected_success=res.success)

    if nominal:
        from horizontal_nominal_audit import audit, verify_reference
        replay = audit(trial_dir)
        save(trial_dir / "nominal_replay.json", replay)
        verify_reference(replay, res.success)
        verified["nominal_replay"] = replay

    dest = DATA / "complete_sim" / bucket / stamp
    save(staged_ep / "verification.json", verified)

    payload = {
        "episode_id": stamp,
        "installed_path": str(dest.relative_to(DATA)),
        "success": res.success,
        "perturbation": perturbation,
        "verification": verified,
        "episode_summary": asdict(res),
        "outcome_sha256": digest(staged_ep / "outcome.json"),
    }
    if nominal:
        trajectory = trial_dir / "nominal_reference.jsonl"
        with (trial_dir / "trajectory.jsonl").open() as source, trajectory.open("w") as commands:
            for i, line in enumerate(source):
                row = json.loads(line)
                command = dict(t_s=i / 30, phase=row["phase"], target_14_si=row["action"])
                commands.write(json.dumps(command) + "\n")
        payload["reference_trajectory"] = str(trajectory.relative_to(DATA))
        payload["trajectory_sha256"] = digest(trajectory)
        payload["reference_layout"] = "30 Hz interval-start targets. Left J1-J6 rad, width m, right J1-J6 rad, width m."
        payload["scene_profile_sha256"] = runner.nominal_profile_sha256
        payload["artifacts_sha256"] = {str(path.relative_to(staged_ep)): digest(path)
                                       for path in sorted(staged_ep.rglob("*")) if path.is_file()}
    manifest = ROOT / "experiments/closed_loop_v1" / ("nominal_horizontal_v2.json" if nominal else f"recorded_{stamp}.json")
    pending = manifest.with_suffix(f".{stamp}.tmp")
    save(pending, payload)
    staged_ep.rename(dest)
    pending.replace(manifest)
    shutil.rmtree(stage_root)
    print(json.dumps({"episode_id": stamp, "installed": str(dest), "success": res.success, "frames": res.frames}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "benchmark", "record-episode", "record-nominal"), default="benchmark")
    parser.add_argument("--strip-xy-mm", type=float, nargs=2, default=[6.0, -4.5])
    parser.add_argument("--strip-yaw-deg", type=float, default=1.2)
    parser.add_argument("--holder-xy-mm", type=float, nargs=2, default=[-4.0, 3.5])
    parser.add_argument("--inhand-xy-mm", type=float, nargs=2, default=[1.5, -1.2])
    parser.add_argument("--extrinsic-bias-xy-mm", type=float, nargs=2, default=[1.5, -0.8])
    parser.add_argument("--open-loop", action="store_true")
    args = parser.parse_args()

    pert = {
        "strip_xy_mm": list(args.strip_xy_mm),
        "strip_yaw_deg": float(args.strip_yaw_deg),
        "holder_xy_mm": list(args.holder_xy_mm),
        "inhand_shift_xy_mm": list(args.inhand_xy_mm),
        "extrinsic_bias_xy_mm": list(args.extrinsic_bias_xy_mm),
    }

    if args.mode == "benchmark":
        summary = run_benchmark(use_learned_policy=True)
        print(
            f"\nSUMMARY: Open-Loop {summary['open_loop_successes']}/{summary['conditions_tested']} -> "
            f"Closed-Loop Ep1 {summary['closed_loop_ep1_successes']}/{summary['conditions_tested']} "
            f"(avg probes {summary['mean_probes_ep1']}) -> "
            f"Self-Evolved Ep2 {summary['closed_loop_ep2_successes']}/{summary['conditions_tested']} "
            f"(avg probes {summary['mean_probes_ep2']})"
        )
    elif args.mode == "record-nominal":
        record_formal_episode({}, nominal=True)
    elif args.mode == "record-episode":
        record_formal_episode(pert)
    else:
        policy = load_deployed_residual_policy(RESIDUAL_NPZ)
        sup = OnlineSupervisor(bias_path=DATA / "self_evolution/closed_loop_v1/sim_calibration_bias.json")
        runner = DualArmSimClosedLoopRunner(learned_policy=policy, supervisor=sup, folded_entry=True)
        try:
            res = runner.run_episode(perturbation=pert, closed_loop=not args.open_loop)
            print(json.dumps(asdict(res), indent=2))
        finally:
            runner.close()


if __name__ == "__main__":
    main()

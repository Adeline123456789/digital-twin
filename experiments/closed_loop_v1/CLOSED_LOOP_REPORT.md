# Unified 4-Stage Closed-Loop Plug-and-Socket Pipeline Report (`feat/closed-loop-plug-insert`)

## 1. Consolidated Architecture (Zero 6-Axis Force Sensor, RGB-Only)

- **Embodiment**: 14-DoF dual-arm MuJoCo scene (`left` folded at `[0,0,0,0,0,0, 0.060]`, `right` active on `can_arm2` / USB `1-10.3:1.0`) with **40 mm horizontal pre-grasp opening** (`grasp_board_relative_yaw_deg = 0.0`) so the neighboring black charger is never touched.
- **Stage 0 (`robot_agent/scene_retarget.py`)**: Unobstructed external RGB subpixel ECC + semantic ray-plane alignment at world socket height `Z = 0.0533 m` and plug housing height `Z = 0.0720 m`, evaluated directly at `outlet3_px` and `plug_center_px`.
- **Stage 1 (`robot_agent/wrist_inhand.py`)**: Post-lift right-wrist RGB yellow silhouette + gripper encoder width retention gate & in-hand offset estimator (`16 mm @ 1.0 N·m` positive lock on the real plug's `~26 mm` molded side flats; `27 mm` in sim).
- **Stage 2 (`robot_agent/compliant_residual.py`)**:
  - Pure-NumPy deployable 499-D `NumpyResidualMLP` (`15 × 33` proprioception history + `4` wrist relative features, `val_mae_mm = 0.6214`, zero privileged inputs).
  - `SmoothResidual4D` (`omega = 7.5 rad/s`, strictly bounded `±4.2 mm` XY, `<= 24 mm/s`) + self-calibrating free-air servo lag baseline + zero-drag vertical lift to `25.0 mm` before sequential `1.1 mm`-chord ring probing.
- **Stage 3 (`OnlineSupervisor` in `robot_agent/bimanual_closed_loop.py`)**: Consolidates winning residual XY offsets into persistent calibration bias without rebuilding the MuJoCo digital twin or invoking runtime LLMs.

## 2. Simulation Verification Summary

- **Unit & Hardware Dry-Run Safety Tests (`tests/test_closed_loop_pipeline.py`)**: `5/5 PASS` (`<= 12 deg/s` speed cap, joint limits, left-arm invariance, empty-grasp rejection, persistent supervisor memory).
- **Paired 6-Condition Perturbation Benchmark (`experiments/closed_loop_v1/benchmark_results.json`)**:
  - Open-Loop Baseline: **1/6** (`C01` only; fails on all scene shifts & extrinsic biases).
  - Closed-Loop Episode 1 (Cold Start): **5/6** (`0` probes on `C01`, `C02`, `C03`; `9` probes on `C05` `+1.8/-0.8 mm` unmodeled extrinsic bias; `5` probes on `C06` combined full challenge).
  - Self-Evolved Episode 2 (`OnlineSupervisor`): **5/6** with **0 probes** on `C01, C02, C03, C05, C06` (one-trial self-evolution absorbed unmodeled extrinsic bias into nominal pre-insert alignment).
- **Validated 3-Camera Formal Simulation Episode**: Installed and verified via `episode_format.validate` at `complete_sim/succeed/20261125T062625Z_62edfb16/` (`2180` frames across `cam_external`, `cam_left_wrist`, `cam_right_wrist`).

## 3. Live Real-Robot Closed-Loop Execution (`can_arm2`) & Calibrated Fix

1. ** Stage 0 Live RGB Repeatability**: Measured tabletop shift relative to Nov 16 reference (`0000.jpg`) across two consecutive runs as `strip_dxy = [+0.15, +4.84] mm, +0.30°` and `[+0.17, +4.85] mm, +0.31°` (`0.02 mm` repeatability, `ECC cc = 0.9918`), `holder_dxy = [-2.17, +2.51] mm` (`ECC cc = 0.9972`).
2. **Grasp & Transport Verification (`20261125_064727_closed_loop`)**:
   - `40 mm` horizontal grasp cleared the black charger with `> 12 mm` margin and locked cleanly into the yellow plug's molded side recesses (`gripper_width = 26.20 mm`, `yellow_area_px = 30,240`, `0.0 mm` slip throughout transport and probing).
   - `stage2_seated_wrist.png` confirms the yellow plug was positioned dead-center over **outlet 3** of the fixed white power strip.
   - CAN telemetry (`commanded_probe.csv`) revealed that normal free-air dynamic tracking lag on real `can_arm2` (`SDK 15%`) reaches `+1.90 mm` (vs `+0.18 mm` in MuJoCo), causing the `0.48 mm` sim threshold to false-trigger in free air at `z = 96.5 mm` (`19 mm` above seated bottom) before the prongs descended to touch the socket. All 6 motors + gripper automatically disabled (`any_enabled = False`) after folding to `[0,0,0,0,0,0]`.
3. **Applied Real-CAN Fix (`scripts/execute_real_closed_loop.py`)**:
   - Raised real-hardware `contact_deflection_threshold_m` to `2.80 mm` active only at `cmd_h <= 16.5 mm` (cleanly above the `1.90 mm` free-air CAN lag ceiling while well below the `3.0°` SDK tracking limit).
   - Added a **no-midair-release guard** (`abort_retreat_held`) so the gripper never opens above the table if seating depth is not reached.

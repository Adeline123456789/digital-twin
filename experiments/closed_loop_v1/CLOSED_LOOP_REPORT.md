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
   - In `stage2_seated_wrist.png` the plug overlaps the outlet and blue switch region. It occludes the pin holes. This frame alone cannot establish metric centering or seating.
   - CAN telemetry (`commanded_probe.csv`) revealed that normal free-air dynamic tracking lag on real `can_arm2` (`SDK 15%`) reaches `+1.90 mm` (vs `+0.18 mm` in MuJoCo), causing the `0.48 mm` sim threshold to false-trigger in free air at `z = 96.5 mm` (`19 mm` above seated bottom) before the prongs descended to touch the socket. All 6 motors + gripper automatically disabled (`any_enabled = False`) after folding to `[0,0,0,0,0,0]`.
3. **Applied Real-CAN Fix (`scripts/execute_real_closed_loop.py`)**:
   - Raised real-hardware `contact_deflection_threshold_m` to `2.80 mm` active only at `cmd_h <= 16.5 mm` (cleanly above the `1.90 mm` free-air CAN lag ceiling while well below the `3.0°` SDK tracking limit).
   - Added a **no-midair-release guard** (`abort_retreat_held`) so the gripper never opens above the table if seating depth is not reached.


## 4. RGB-registered horizontal nominal, 25 November

At 16:19 China Standard Time this robot’s external camera supplied a fresh color frame. Four blue socket panels were resolved in that frame, including AC3 at (336.60, 266.89) px. Intersecting the four rays with the socket face and fitting the known panel pitch gave the strip center (0.316659, 0.020935) m and yaw −89.902°. The largest panel residual was 1.22 mm. The uncovered holder slot at (477.43, 265.95) px fixed the holder translation on its CAD top plane at world Z = 0.058 m. Applying the small RGB rotation since the morning capture to the specified −89.5° yaw anchor gave a holder center of (0.296204, −0.358950) m and yaw −89.512°. These estimates depend on this robot’s existing RGB camera calibration.

In the morning wrist images the tips bracket the narrow rear of the plug and the housing obscures the third outlet during insertion. The simulated wrist video from `20261123T092135Z_48f4ff46` and the authorized `000210` wrist video show the desired shoulder engagement. The new nominal opens to 40 mm, commands the grasp center 16 mm above the plug face and shifts it 4 mm toward the shoulder in holder-local +Y. An 8 mm offset away from the charger and a holder-relative XYZ rotation of (4°, −10°, 16°) give the fingers clearance during descent. The 8 mm offset is a command parameter. Surface clearance is measured independently during replay.

The compiler measures how the in-hand pose changes during a simulated carry and fits the tool-to-plug transform at settled waypoints. The plug is straightened in free space before pin entry. It traverses the socket at decreasing height, opens and retreats before folding. The exported reference contains only the 14 commanded positions, phase and simulation time. The replay uses those commands without access to object poses for control. No robot motion commands were sent.

From the repository root, `../piper_simulation/.venv-mujoco-hardware/bin/python scripts/run_plug_closed_loop.py --mode record-nominal` generates a new simulation reference using the saved RGB registration. The measured clearances enclose each modeled rigid body's collision surfaces in a convex hull. The charger comparison includes both arms and the carried plug. The holder comparison includes both arms and permits the plug to sit in its holder.

The final grasp frame contains contacts on both molded shoulder surfaces at plug-local heights of 12.88 mm and 7.66 mm. The commanded J6 angle is −10.49° at closure and −13.76° after lift. At the end of transport J6 is +40.13° and the signed world-X displacement of the plug from AC3 is +0.451 mm. After alignment the displacement is −0.021 mm. Following release, retreat and folding it is −0.230 mm, with a final XY distance of 0.234 mm. The real wrist views establish the direction of the original discrepancy. These numerical residuals describe the registered simulation.

| Formal episode `20261125T083827Z_a71d18d9` | Verified result |
| --- | ---: |
| Frames per external, left-wrist and right-wrist RGB stream | 2,201 |
| Physics substeps replayed from the folded initial state | 281,728 |
| Peak target / measured right-joint speed | 9.9983 / 11.4830 °/s |
| Peak target / measured J5 during transport, preinsert, alignment and insertion | 63.786 / 63.787° |
| All-contact maximum penetration | 0.03739 mm |
| Minimum charger clearance, conservative lower bound | 8.310 mm |
| Minimum arm-to-holder clearance, conservative lower bound | 6.318 mm |
| Native seating after folding | True |
| Recorded / independently replayed joint-state and plug-position maximum difference | 0 SI / 0 m |

The first and last right-arm joint commands are exactly six zeros. Left-arm targets hold six zeros and a 60 mm opening throughout. `episode_format.validate(..., expected_success=True)` verified the native outcome, schema, units, video frame counts and recorded trajectories. Recompiling from the checked-in profile reproduced all 2,201 commands exactly. The local reference JSONL, episode files and SHA-256 digests are listed in [the nominal manifest](nominal_horizontal_v2.json).

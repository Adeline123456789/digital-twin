# Task Specification: Dual-Arm 40 mm Horizontal-Grasp Nominal Posture & Trajectory Refinement

## 1. Context & Why Real Grasp Looked Wrong (`20261125_064727_closed_loop`)

On `2026-11-25`, `feat/closed-loop-plug-insert` validated the 4-stage closed-loop architecture in MuJoCo (`Open-Loop 1/6 -> Closed-Loop 5/6`, formal 3-camera Episode `complete_sim/succeed/20261125T062625Z_62edfb16`) and executed two live trials on `can_arm2` (`1-10.3:1.0`, auto-disabled cleanly).

Visual comparison between:
- Simulated horizontal grasp (`complete_sim/succeed/20261123T092135Z_48f4ff46/videos/cam_right_wrist_rgb.mp4`),
- Allowed RGB posture reference (`REFERENCE_DATA/observations/20260908/000210/videos/cam_right_wrist_rgb.mp4` — RGB posture reference only; never transfer H5 joints or extrinsics), and
- Today's real right-wrist captures (`self_evolution/closed_loop_v1/real_runs/20261125_064727_closed_loop/stage1_wrist_lift.png` and `stage2_seated_wrist.png`)

identifies **three exact geometric discrepancies** explaining why the real gripper did not grasp the yellow plug in the right posture:

1. **Shallow Z Grasp & Rearward (`-X_world`) Pinch on Cord Tail (`~10–12 mm` offset)**:
   - In the target posture (`000210` & `20261123T092135Z_48f4ff46`), the two finger tips advance past the narrow waist up to the wide circular shoulder flange (`grasp_height_m` deeper on the real plug body), fully engaging the flat pads around the plug's center of mass.
   - On real `can_arm2`, `stage1_wrist_lift.png` shows the fingers pinched `~10–12 mm` too high (`+Z`) and `~12 mm` too far toward the rear cord-exit tail (`-X_world`), gripping only the narrow rear neck (`measured width = 26.2 mm` vs `32.8 mm` simulated smooth cylinder).
2. **Legacy Single-Arm Vertical-Grasp Calibration Baked into `holder_xy_m` & `holder_yaw_deg`**:
   - `20261116T105302Z_d700416b/task.yaml` calibrated `holder_xy_m = [0.3035, -0.3653]` and `holder_yaw_deg = -83.74°` using a `J6 = +43°` (`Rz(+90°)`) vertical grasp.
   - In today's dual-arm tabletop (`stage0_external.png`), the fixed grey holder is nearly parallel to the white strip (`holder_yaw_deg ≈ -89.5°`, not `-83.74°` — visible as a `~6°` skew between jaws and grey holder in `stage1_wrist_lift.png`).
   - Rotating `J6` by `-90°` from vertical (`+43°`) to horizontal (`-14°`) reversed the unmodeled wrist/TCP offset vector, shifting the grasp rearward (`-X`) at the holder and then swinging the held plug `+12–15 mm` too far forward (`+X`, over the blue power switch button in `stage2_seated_wrist.png`) when `J6` rotated back to `+41°` over outlet 3.
3. **Missing Explicit Folded-Zero Entry in `20261123T092135Z_48f4ff46`**:
   - `20261123T092135Z_48f4ff46` started at the IK `approach` pose (`initial_robot_pose: "IK approach pose; initialization only"`) rather than the physical folded-zero entry `[0, 0, 0, 0, 0, 0]` required by `Hardware.connect(..., folded_entry=True)`.

---

## 2. Concrete Implementation Deliverables (No Blocking `NO-GO` Gates)

Build directly on branch `feat/closed-loop-plug-insert` without inventing new blocking research gates or copying `piper_simulation` source/assets/raw media into Git:

### Deliverable A: Dual-Arm Scene & Grasp Geometry Alignment
1. Use the **14-DoF dual-arm scene** (`active_arms: [left, right]`, `left` arm strictly `[0, 0, 0, 0, 0, 0, 0.060]`, `right` arm active).
2. Register the current dual-arm tabletop fixture poses against `self_evolution/closed_loop_v1/real_runs/20261125_064727_closed_loop/stage0_external.png` and `stage1_wrist_lift.png` (RGB only; never use `depth.npz`):
   - Correct `holder_yaw_deg` (near `-89.5°`) and `holder_xy_m` / `strip_xy_m` for the **horizontal grasp (`J6 ≈ -10°..-14°`)** so wrist-rotation lever-arm error does not shift the plug toward the blue button at `J6 ≈ +40°`.
   - Refine horizontal grasp depth (`grasp_height_m` ~ `0.012–0.016 m` above plug face, collision-checked against the grey holder top `Z = 0.055 m`!) and forward `+X` grasp center (`+4 to +8 mm` toward plug front shoulder) so the simulated and real `cam_right_wrist` views match `20261123T092135Z_48f4ff46` (`finger tips reaching the wide yellow shoulder, plug centered over the 5-hole socket rather than the power switch`).
   - Keep **`initial_right_opening_m = 0.040`** (`40 mm` pre-grasp opening) and verify `>= 8 mm` 3D clearance to `task/black_charger` across all `approach -> descend -> grasp -> lift` substeps.

### Deliverable B: Verified Folded-Entry Nominal Simulation Episode & Reference Export
1. Generate a complete, collision-free (`max_penetration_m <= 0.0002`), 14-DoF nominal trajectory starting and ending at **explicit folded zero** (`right_deg = [0, 0, 0, 0, 0, 0]`, `initial_right_opening_m = 0.040`, `speed <= 12 deg/s`):
   `prepare_gripper_40mm -> approach -> descend -> grasp -> lift -> transport -> align -> insert -> release -> retreat -> fold`
   (ensuring `J5 <= 68.0°` with smooth carry-tilt taper so hardware `70.0°` `J5` bounds never saturate).
2. Replay inline through `piper_sim` physics, pass `scripts/episode_format.py` `validate(...)`, and install one formal 3-camera Episode under `/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug/complete_sim/succeed/<new_episode_id>/`.
3. Export the nominal reference manifest + trajectory JSONL (or plug directly into `HorizontalGraspNominalPlanner` in `robot_agent/bimanual_closed_loop.py` and `scripts/execute_real_closed_loop.py`) and record hashes in `experiments/closed_loop_v1/nominal_horizontal_v2.json`.

### Deliverable C: Plug Back into Closed-Loop Pipeline & Push to GitHub
1. Verify `tests/test_closed_loop_pipeline.py` and `scripts/execute_real_closed_loop.py --dry-run` pass (`hardware_commands == 0`, `within_joint_limits == True`, `max_joint_speed_deg_s <= 12.0`).
2. Do **not** run real hardware automatically if unattended; commit only clean code, config/manifest JSONs, and reviewed report to `feat/closed-loop-plug-insert` (keep raw recordings, videos, XML, and weight binaries under `/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug/`), update `evolution/current_state.json`, and push to `origin`.

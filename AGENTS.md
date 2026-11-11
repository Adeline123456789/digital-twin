# Project rules

- This repository is an additive robot learning/experiment layer. The existing piper_simulation project is separate: do not copy or upload its source, models, XML, assets or raw recordings here. Use its existing local interfaces to collect data. Public Git contains only new learning code, protocols, reviewed reports and dataset references/hashes.
- Dataset root: `/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug/`.
- Successful simulation video root: `complete_sim/succeed/20261111_sim_calibrated_pick_insert_000001/videos/`. It already contains three videos; never overwrite them. New runs use unique experiment-ID subdirectories within that video root, and matching metadata subdirectories in the same episode's proprio_stats/.
- Failed simulation data: `complete_sim/failed/`, each attempt in a unique directory.
- Successful real data: `complete_real/successed/` (preserve spelling).
- Failed real data: `complete_real/failed/`; use this as the historical real-failure source.
- Expert successful real data: `/media/midea/F496AC8096AC4548/act_td3/connector_plugin/`. It comes from other arms of the same model, with different camera angles and arm spacing. Do not directly replay those joint poses or treat its extrinsics as this robot's calibration. Keep provenance separate.
- This robot has NO six-axis force/torque sensor. Never invent force/torque observations. Gripper motor feedback and simulated contact quantities are not measured six-axis force.
- Real movement requires fresh explicit permission, validated simulation and existing safety checks. Do not change or bypass hardware safety rules.

# 20261112T071105Z_6a45b3cd

- **task**: "Stage 3 round 1 current original-profile recording baseline"
- **environment**: "simulation"
- **result**: "success"
- **task_success**: true
- **validation_status**: "SIMULATION_VERIFIED"
- **failure_signature**: null
- **root_cause**: "UNKNOWN"
- **root_cause_status**: "unknown"
- **execution_time**: 105.2407561299915
- **tokens**: null
- **lessons_learned**: ["Outcome applies only to this simulation configuration; real cause remains UNKNOWN."]
- **next_improvement**: "Compare matched initial states and independent holdout shifts before accepting repair"

See result.json for provenance and evidence hashes. Raw data stays in the existing dataset.

Stage 3 baseline: uses simulator privileged object/gripper poses, collision penetration and task-success truth. Existing simulation has NOT passed independent real calibration. Headless success is NOT yet a valid recorded Episode; recording and verification are pending. Original-profile 12 deg/s and existing smooth timing retained without additional slowdown or smoothing.

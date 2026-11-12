# 20261112T071528Z_0d05e556

- **task**: "Three-view physics replay 20261112T071105Z_6a45b3cd"
- **environment**: "simulation_recording"
- **result**: "success"
- **task_success**: true
- **validation_status**: "SIMULATION_VERIFIED"
- **failure_signature**: null
- **root_cause**: "UNKNOWN"
- **root_cause_status**: "unknown"
- **execution_time**: 1.2225695390079636
- **tokens**: null
- **lessons_learned**: ["Videos decoded and checked against metadata; recordings are simulation, not real robot evidence."]
- **next_improvement**: "Review visible contact and seating; obtain independent real pose evidence before calibration"

See result.json for provenance and evidence hashes. Raw data stays in the existing dataset.

Stage 3: privileged simulator state/action, object pose, collision penetration and task-success truth used for verification. All three videos have 1133 frames, state/action arrays and object replay match the source trial. This is one valid simulation success, not a real-calibration result.

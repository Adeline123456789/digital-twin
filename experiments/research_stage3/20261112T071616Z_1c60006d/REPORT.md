# 20261112T071616Z_1c60006d

- **task**: "Stage 3 round 1 historical failure reproduction and recording"
- **environment**: "simulation"
- **result**: "failure"
- **task_success**: false
- **validation_status**: "SIMULATION_FAILED"
- **failure_signature**: "Released plug did not meet stable insertion criteria"
- **root_cause**: "UNKNOWN"
- **root_cause_status**: "unknown"
- **execution_time**: 89.81350620498415
- **tokens**: null
- **lessons_learned**: ["Outcome applies only to this simulation configuration; real cause remains UNKNOWN."]
- **next_improvement**: "Compare matched initial states and independent holdout shifts before accepting repair"

See result.json for provenance and evidence hashes. Raw data stays in the existing dataset.

Privileged simulator object/gripper poses, collision penetration and task-success truth were used. This is the predeclared historical +2 mm insertion-offset failure control; the zero-offset baseline succeeded. All motion/physics settings retained. Failure recording pending; no real-robot or calibration claim.

Recording now verified: experiments/evaluation/20261112T071958Z_4779a01a; 1130 frames per camera, failed physical outcome preserved.

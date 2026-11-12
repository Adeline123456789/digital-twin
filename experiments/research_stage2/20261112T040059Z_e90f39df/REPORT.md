# 20261112T040059Z_e90f39df

- **task**: "Existing table/camera fit with separate real-episode validation"
- **environment**: "real_sim_calibration"
- **result**: "success"
- **task_success**: null
- **validation_status**: "CALIBRATION_REJECTED"
- **failure_signature**: null
- **root_cause**: "UNKNOWN: independent plane residual does not identify real insertion failure cause"
- **root_cause_status**: "unknown"
- **execution_time**: 10.36268883300363
- **tokens**: {"planner": 0, "research_orchestrator": null}
- **lessons_learned**: ["Only three camera/table degrees of freedom are observable from a plane; world XY/yaw remain CAD assumptions"]
- **next_improvement**: "Use only validated observable geometry; task-cost benefits require separate simulation tests"

See result.json for provenance and evidence hashes. Raw data stays in the existing dataset.

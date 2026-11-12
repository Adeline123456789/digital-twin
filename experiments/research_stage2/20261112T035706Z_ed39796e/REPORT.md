# 20261112T035706Z_ed39796e

- **task**: "RGB-D evidence availability at grasp/lift transitions"
- **environment**: "real_historical"
- **result**: "success"
- **task_success**: null
- **validation_status**: "OBSERVABILITY_AUDIT_NOT_CALIBRATION"
- **failure_signature**: null
- **root_cause**: "UNKNOWN; RGB/depth availability alone does not determine contact physics"
- **root_cause_status**: "unknown"
- **execution_time**: 6.975711299019167
- **tokens**: {"planner": 0, "research_orchestrator": null}
- **lessons_learned**: ["Nonzero depth is not a verified object surface; missing/occluded data cannot be replaced with simulator ground truth"]
- **next_improvement**: "Verify actual surface correspondences and camera geometry before accepting any physical fit"

See result.json for provenance and evidence hashes. Raw data stays in the existing dataset.

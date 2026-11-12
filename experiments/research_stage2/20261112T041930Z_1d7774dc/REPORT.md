> Superseded evidence status: SUSPENDED_UNVERIFIED_DEPTH_PROVENANCE. User reports no real RGB-D. The original labels below are not proof of sensor provenance; see ../round5/REPORT.md. No real calibration claim is accepted.

# 20261112T041930Z_1d7774dc

- **task**: "Cross-view object and finger geometry validation"
- **environment**: "real_historical"
- **result**: "success"
- **task_success**: null
- **validation_status**: "GEOMETRY_AUDIT_ONLY"
- **failure_signature**: null
- **root_cause**: "UNKNOWN: camera/FK, occlusion, surface correspondence and depth errors remain possible"
- **root_cause_status**: "unknown"
- **execution_time**: 8.387874546984676
- **tokens**: {"planner": 0, "research_orchestrator": null}
- **lessons_learned**: ["Projected surface agreement is a necessary consistency check, not a full object pose or calibrated aperture"]
- **next_improvement**: "Stop physical fitting if geometry evidence fails; retain raw observations and numerical audit"

See result.json for provenance and evidence hashes. Raw data stays in the existing dataset.

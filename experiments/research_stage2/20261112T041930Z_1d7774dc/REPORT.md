> **用户最终确认（2026-11-12）：历史和当前真机视觉均只有 RGB。** 深度文件明确排除出本机器人真机测量证据；第三、四、五轮依赖深度的真实几何解释不成立，不再标为等待用户确认。文件实际来源仍未知，保留审计记录，不猜测其生成方式。仅保留文件上的数值分析，不支持真实物理校准；不影响独立于深度的关节/夹爪分析。后续方法必须适配 RGB 和已确认的运动反馈。

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

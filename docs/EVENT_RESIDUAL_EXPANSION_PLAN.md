# Conditional expansion after native and frozen replay gates

This is a simulation-only plan for a dual-arm model executing the existing right-arm task. True dual-arm coordination is a separate future study.

1. Require the specified unmodified zero-offset baseline, native success detector and full Episode validator to pass. Export a hash-bound reference and replay it through native physics without invoking IK/planner or exposing privileged state to the policy. Stop on failure.
2. Implement the five-state comparator, Strong Kleene predicates, clock/visibility/confidence handling and UNKNOWN no-action checks. Keep privileged ground truth separate from RGB/proprioception input.
3. Validate interpretable RGB plug/socket features. Stop with observation contract failure if they cannot support the event predicates. Add the minimum right-joint residual adapter only after a real simulation injection check, preserving left reference actions and native bounds.
4. Pilot static socket X shifts at -2/-1/0/+1/+2 mm. Do not confuse the existing collector's **holder** `--shift-mm` with socket perturbation. Static retarget is not event-gating innovation evidence. Omit optional VS002 Oracle for this batch.
5. Measure collision geometry clearance, preregister three transient socket-Y amplitudes and a fixed post-preinsert window, and verify movement in collision geometry and RGB. Freeze paired pilot/held-out initial-state seeds, the shared control law, bounds, event map and thresholds before confirmation. Merely varying a seed while randomization is disabled does not create independent conditions.
6. Compare nominal/ungated/phase-gated/event-gated policies over at least 30 paired conditions, reserving independent held-out seeds. Use paired bootstrap intervals. Expand to 50+ only for declared statistical precision, never to tune toward a positive conclusion. Test occlusion, OOD and incompatible legacy contracts separately.
7. GO requires every taskbook gate, including supported paired benefit, event precision, false-trigger bound and zero forbidden execution. Unsupported intervals are INCONCLUSIVE. A prerequisite/interface/detection failure is NO-GO and ends this batch. No PPO, new agents, memory expansion, contact tuning or Twin rebuilding follows a NO-GO.

The hardware-control import regression cannot run under this taskbook's `real_data_read`/control-root prohibition; report that limitation explicitly. Raw data, simulator sources/assets, XML and videos stay outside Git. Publish only new code, reviewed reports, relative evidence paths and hashes on `research/event-residual-dual-20261123`.

## Batch outcome and user amendment

The 2026-11-23 batch stopped at Stage 3 with **NO-GO: observation contract failure**; see `experiments/event_residual_mvp/MVP_REPORT.md`. The original frozen physical replay passed, but OpenCV V0 did not qualify target socket identity. Steps 4–7 above remain unexecuted future work, not automatic continuation instructions.

The user subsequently authorized only the expert `000210` RGB videos as posture reference and suggested about 40 mm initial opening. A separate own-IK simulation candidate completed the task, and its 146-frame grasp prefix had zero black-charger/right-arm contacts over 18,688 checked integrator endpoints. No expert H5, joints or camera extrinsics were used. This candidate did not replace the frozen reference and did not resolve the V0 observation failure.

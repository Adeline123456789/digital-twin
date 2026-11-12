# Research restart — 2026-11-12

Baseline: `cc17a89a5fc611747ee6d7aa6232aaa2a1fc3851`, clean working tree; remote HEAD matched before this round. This report supersedes the earlier workflow-completion wording for the new four-stage research objective. No stage is passed merely because scripts exist.

## Filesystem and provenance

The requested simulator and expert-data directories exist. The requested real-failure directory contains four episodes. The two simulation paths in the research text contain `8374e8e3d3e` and do **not** exist. The previously user-authorized, filesystem-verified dataset root is `/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug`. This is an explicit correction based on filesystem evidence, not silent substitution.

Its formal simulation failures now contain three episodes: the two original `sim_real05` recordings and `20261111T083720Z_f64309a8`. Fourteen source-only episode directories were deleted at the user's request in the baseline commit. Their historical result reports and separate raw trial evidence remain; they must not count as available formal episodes. Successful recordings use the existing unique-ID children under the designated `20261111_sim_calibrated_pick_insert_000001` episode. A successful headless trial alone is not a valid recorded episode.

There are two different simulator sources: the original `piper_simulation` Git checkout and the retained `calibration/offline_workspace_20261110` snapshot with its documented 25 deg/s free-space profile. They are not interchangeable, and their success denominators must remain separate. Both remain outside public Git. The expert dataset comes from another installation; it is not this robot's calibration or validation set.

## Actual policy and remaining gaps

- `evolution/active.json` is v2. `learn.decide()` reads it, but `learn.simulate()` calls `decide()` only when given a failure signature.
- `agent_loop.run()` calls `simulate()` without a signature. Consequently its actual A/B/C/D behavior is selected by `--arm`, with D's repeat check and feedback implemented directly in that runner; it is **not** execution of the active v2 gate. `agent_challenge.py` is a separate challenge runner. Version manifests are not proof of active deployment.
- Existing feedback experiments demonstrate correction of two signed offsets in one simulated task family, not generalization to new task families. Outlet 1 transport IK failures remain unresolved. The action interface there exposes only insertion offset, which does not address earlier transport planning.
- The 20k input-token cost is observed CLI usage, not a hypothetical estimate. Existing easy-task comparisons show no benefit. Total cost per valid recorded success and sustained throughput remain unmeasured.
- Historical phase/trajectory and wrist-image analyses exist, but global motor RMSE does not localize the first physical divergence. Neither real object 3D trajectories nor six-axis force measurements are available from those analyses. Existing SDK effort is not calibrated contact force.
- No independently validated physical calibration has been accepted. The retained scene contains candidate parameters, not measured ground truth.

No CodeGraph directory or codebase-memory MCP tools are available in this session; the task-directed audit used current source and callers. External original Agent orchestration remains outside the inspected additive runner.

## Stage 1, round 1 decision

Use ASPIRE's trace-guided diagnosis mechanism, not its simulator stack: summarize action/effect evidence by existing phase, separate API status from physical outcome, and inspect only relevant numerical traces. Reuse `compare_replays.py` and existing JSONL/HDF5 recordings. Compare against its previous global-RMSE-only output on the same historical executions. Add simulated object/contact summaries explicitly marked privileged. The first experiment is offline analysis: no new task rollout, controller changes, speed changes, or hardware calls.

Budget: zero additional planner-model calls; CPU trace analysis. Research/orchestrator tokens are not independently metered and will remain null rather than called zero. Stage advancement requires a measured discrepancy with provenance and a falsifiable explanation; unique physical causes may remain UNKNOWN. Subsequent rounds depend on this result. Stage 2 must pass its independent-validation gate before Stage 3 can start.

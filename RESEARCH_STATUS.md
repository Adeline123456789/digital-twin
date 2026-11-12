> 用户最终决定（2026-11-12）：**保持六轮上限，保留未完成结论。** 不追加第七轮，不进入尚未满足前提的 Stage 3/4。现有结果全部保留；本决定不是新的实验，也不是目标完成声明。

> 最新状态（2026-11-12）：用户已授权并完成第六轮 RGB 离线研究。三条记录的测量门槛均未通过，未进入拟合；Stage 2 已用六轮，Stage 3/4 未启动。后文是历史审计，当前结论以 FINAL_RESEARCH_REPORT.md 和 experiments/research_stage2/round6/REPORT.md 为准。

> **用户最终确认（2026-11-12）：历史和当前真机视觉均只有 RGB。** 深度文件明确排除出本机器人真机测量证据；第三、四、五轮依赖深度的真实几何解释不成立，不再标为等待用户确认。文件实际来源仍未知，保留审计记录，不猜测其生成方式。仅保留文件上的数值分析，不支持真实物理校准；不影响独立于深度的关节/夹爪分析。后续方法必须适配 RGB 和已确认的运动反馈。

> **2026-11-12 用户纠正，优先于下文历史描述：** 实际真机没有 RGB-D 图像信息。历史目录虽有 depth.npz，其传感器来源未核实。第三、四轮所有依赖深度的真实几何解释暂停使用；数值仅保留为文件分析记录，不作为真机实测或独立物理验证。获准追加的第五轮已执行，8 对检查为 0/8，通过与否均不能解决来源问题；无校准晋升。Stage 2 共五轮仍未通过，Stage 3/4 未启动。详见 experiments/research_stage2/round5/REPORT.md。下文“四轮”等为追加研究前的历史总结。

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

## End-of-budget update

Stage 1 passed its measured-discrepancy gate in one round. Stage 2 used four adaptive rounds and did not pass its independently validated improvement gate. Timing, constant aperture offset and table-registration candidates were not promoted. Stage 3 and Stage 4 remain NOT RUN, prerequisite unmet. See FINAL_RESEARCH_REPORT.md for the gated NO-GO conclusion, actual measurements, commit links, and unfinished work. The overall research objective remains unmet; no background hardware or experiment job is running.

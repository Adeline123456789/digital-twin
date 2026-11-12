> Stage3 第三轮已完成：Luna输入Token 22,034→16,295（−26.0%），Astra 22,500→16,925（−24.8%）。Luna两组均成功；Astra两组均失败，暴露世界/插座坐标混用及接口语义缺口。因此没有切换通用默认。新增2条成功、2条失败完整Episode；没有一致耗时或总费用收益。见 [第三轮报告](experiments/research_stage3/round3/REPORT.md)。下一轮改进观测与探索，Stage4仍待开展。

# 当前进展（2026-11-12，Stage 3 第二轮完成）

Stage 2 按用户要求保持六轮，未通过独立真机校准验证，并已继续后续目标。Stage 3 已完成两轮，Stage 4 尚未开始；整体目标尚未完成。

本轮三组原路径/单次录制对照，各自成功 2/3（含一个预设失败控制，不能视为自然成功率）。完整数值运动、接触、相机参数和原生结果一致；包含失败成本的采集组件耗时/成功 Episode 从 **361.0 秒降至 221.7 秒，减少 38.6%**。六条完整 Episode 已保存；原速度、平滑和数据格式保持。

**Simulation-only，使用并披露仿真特权信息。** 首个逐像素相等门槛未通过；原生重复也不满足该门槛，测量修正已在后续失败/留出场景前公开。视频不逐像素相同；上述结论限于运动、格式和测得的采集组件耗时。总研究费用、Token 收益、弱模型能力、人工干预减少和真实数字孪生校准均未证明。

单次录制以 `simulate(..., record_inline=True)` 显式启用，随后使用原 finalizer；默认 Agent 路径和 active.json 的 v2 保持。版本、回滚、证据和来源见 [第二轮报告](experiments/research_stage3/round2/REPORT.md) 与 [候选版本](evolution/versions/inline_collection_v1.json)。下一轮处理模型上下文开销，并进行实际探索/反馈比较。

**研究判断：整体目标 NO-GO（尚未完成）；此项采集优化为有范围限制的可用候选。**

---

## 以下为先前阶段记录（不代表当前阶段状态）

> Stage 3 第一轮已完成：新增 1 条完整成功 Episode 和 1 条完整失败 Episode，三路视频/状态/动作/物体重放均核对。实际使用并披露仿真特权信息；原速度与物理参数不变。见 [采集基线报告](experiments/research_stage3/round1/REPORT.md)。尚未证明采集成本下降。

> 最新用户决定：Stage 2 保持六轮并收口，未通过独立校准验证；继续 Stage 3/4。后续仿真允许特权信息，报告必须披露，不能声称真实提升或将收益归因于已完成的校准。

> 最新推进（2026-11-12）：针对 RGB 观测缺口复用了 SIFT、LoFTR、ECC 和仓库已有 LK。固定软件变换检查中 LoFTR/ECC 为 1/4、既有 LK 为 3/4；四组历史真机图像保留 28–30 个局部点，并发现局部点只支撑部分图像区域，不能直接外推为全物体测量。38 项测试通过。详见 [RGB 对应与误差定位报告](experiments/diagnosis/20261112T061643Z_rgb_correspondences/REPORT.md)。保留局部观测，未接受新物理校准或声称采集效率提升。

> 最新实现（2026-11-12）：原项目 FK 输入与现成 OpenCV 手眼求解器已接通，33 项软件测试通过。真实多轴素材已准备机器人侧位姿；真实 RGB 目标位姿仍未提供，未执行新标定。详见 [接口验证报告](experiments/evaluation/20261112T060334Z_handeye_adapter/REPORT.md)。这是软件验证，不是 Stage 2 通过或采集效率提升。

> 最新推进（2026-11-12）：按用户要求，针对具体故障核对前人实现并修正代码。已修复右腕图像误用主相机时间轴的问题，28 项测试通过；找到了此前漏检的多轴 RGB/关节标定素材及同型号 Piper 的 Bi-JCR 实现。详见 [修正报告](experiments/diagnosis/20261112T054835Z_rgb_method_correction/REPORT.md)。Stage 2 六轮上限保持，没有新参数实验或校准晋升；整体未完成不等于所有研究路线不可行。

> 先前用户决定（2026-11-12）：**保持六轮上限，保留未完成结论。** 不追加第七轮，不进入尚未满足前提的 Stage 3/4。后续用户要求继续具体方法与实现工作，因此此决定不再被解释为停止所有只读核查和软件修复。

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

# Event-Driven Residual Adaptation MVP — NO-GO

2026-11-23。结论：**NO-GO，失败类别为 observation contract failure**。原生任务、冻结参考物理重放，以及用户补充的 40 mm 视频姿态抓取候选均通过各自的仿真验证；当前 OpenCV V0 未建立可信的目标插孔对应关系，不能开展有依据的视觉 Residual。按任务书在 Stage 3 停止，不进入扰动和四策略对照。

这属于双臂模型中的右臂任务；没有实现双臂协同。全部执行结果为 Simulation-only，不是 Real-verified。

## 已完成的验证

| 检查 | 结果 | 证据 |
|---|---|---|
| Stage 0 原生接口审计 | 14 维 SI 动作；右臂执行，左臂参考保持；原生 generator 存在基于物体/目标位姿的 IK 重规划 | `STAGE_0_AUDIT.md` |
| 原生零扰动基线 | 完整任务成功；三路 RGB 各 1,133 帧；Episode schema、单位和记录一致性通过 | `nominal_manifest.json`；Episode `20261123T090549Z_a35ab76b` |
| Frozen Reference 实际物理重放 | 完整任务和 Episode 验证通过；相对于 generator 的最大关节状态差为 0；执行动作哈希相同 | `stage1_replay.json`；Episode `20261123T091030Z_9c2fbd70` |
| 冻结后的调用/输入边界 | Native `PoseIK.solve` / `main` spy 调用 0；Player 仅持有不可变动作/phase/time，未接收 backend 或 `MjData` | `robot_agent/nominal_reference.py`、对应单测和上述重放报告 |
| Stage 2 事件语义 | 五种比较结果、九事件先后关系、Strong Kleene 真值表、deadline、时钟中断、缺失/低置信观测通过单测 | `stage2_contract_results.json` |
| UNKNOWN 安全传播 | `UNRESOLVED → NO_ACTION`；关键事件 HOLD；即使设置恢复/结构候选标志也不改为动作 | 对应 router 单测及 RGB 审计明细 |
| Stage 3 RGB V0 | 目标插孔身份未通过，联合视觉误差不可用 | 两份 `observation_<episode>.json` |

冻结参考 hash 为 `a433cb513b9513e77badc8edd4b7de648e07f62286e8520c8a40d12d1dac7b18`。Nominal / executed action hash 均为 `7849872c6fca43eb9194dc75b75cd71237b4b14539818e9a572397d23f61ceb3`。这不是离线数组相等冒充执行：重放逐帧调用原生 `apply` 和每个积分步的碰撞检查，生成了新的完整三视角 Episode。

Privileged access 结论来自接口隔离、代码检查和已审计 planner 入口 spy，不是操作系统沙箱证明。原生 generator 本身仍使用特权位姿并重规划；历史采集器中的 `planner_calls=0` 不代表它从未调用 IK。本报告的零重规划结论只针对冻结 Player rollout。

## 用户补充：参考视频姿态，初始开口 40 mm

用户随后明确授权读取 `REFERENCE_DATA/observations/20260908/000210/videos`。只读取了三个 RGB MP4（各 212 帧、30 Hz），检查了帧 0/42/84/127/169/211；没有读取 expert H5、关节轨迹或迁移相机外参。视频仅支持定性的夹持方向参考，40 mm 是用户给定的候选值，不是从视频反算的测量值。视频哈希和授权边界保存在 `grasp_amendment.json`。

独立候选沿 holder 插头排列方向夹持，初始右夹爪宽度 40 mm，闭合目标仍为原生 27 mm、抓取高度仍为 20 mm。当前模型自行求解 IK；未复制示教关节姿态。通过一个局部运行时 adapter 改变 generator 的初始开口和抓取方向，没有修改外部仿真源码、接触参数、成功阈值或原冻结参考。

- 抓取/抬升前缀：146 帧，抬升约 52.7 mm，原生 lift 检查通过，最大穿透约 0.0223 mm。该前缀保留在 `self_evolution/event_residual_mvp/video_grasp_40mm_20261123`，不是正式完整 Episode。
- 同一前缀的独立冻结接触重放：18,688 个刷新后的积分端点，右臂与黑色插头接触计数 **0**，状态重放误差 **0**。这支持“该仿真抓取前缀未观察到碰黑色插头”，不证明真实硬件或所有姿态都不会接触，也不是连续时间数学保证。
- 完整候选 `20261123T092135Z_48f4ff46`：原生插入、释放、稳定就位及完整 Episode 验证通过；最大穿透约 0.0227 mm。详见 `video_grasp_complete.json`。

开口和方向共同变化，不能把效果单独归因于 40 mm。新候选没有替换原冻结参考，也没有完成自己的完整冻结重放/多 seed 重复性验证；它不能直接作为已验证 Residual 实验基线。

## 观测为何未通过

固定 V0 使用 RGB HSV/连通区域：黄色外壳、蓝色面板内的暗孔、外壳附近的细长金属候选。插孔身份规则需要可辨认的 1 个 USB + 3 个 AC 模块排列，以确定 outlet 3；遮挡后不把邻近可见插孔误当成目标。候选针尖不是经过语义验证的真实针尖，置信度没有被提升至事件动作阈值。

| 场景 | 关键时刻数 | 三相机样本数 | 可靠目标插孔身份 | 可用联合相对特征 |
|---|---:|---:|---:|---:|
| 原冻结基线 | 75 | 225 | 0 | 0 |
| 视频姿态 / 40 mm 候选 | 91 | 273 | 0 | 0 |

范围为 transport 终点及每一个 align/insert 帧。498 个相机样本中，474 个在 deadline 后进入 UNRESOLVED；其余处于 PENDING，Residual 和 Structural 路由均为 0。它们是相关视频帧，不是 498 个独立实验条件。

额外可见场景控制帧同样暴露 V0 的分辨率/连通孔计数局限：外部视角的 AC 面板会产生“两孔候选”，因此失败不能全部归因于遮挡。这个结果否定的是**当前 V0 观测实现的适用性**，不是 RGB 方法本身。代码保留明确的 UNKNOWN，不输出未经验证的位姿或目标误差。

独立 evaluator 还恢复了存档仿真状态，使用渲染 geom ID 和模拟 depth 检查可见性：原基线右腕在 75/75 个关键时刻有少量插针表面像素，但只有 2 个时刻有符合 1 mm / 3×3 像素容差的针尖角点；新候选右腕为 3/91，外部为 4/91。表面可见不等于可可靠定位针尖。该特权 evaluator 的输出没有进入 Student；它没有推进物理、移动相机或修改场景。详见 visibility 报告，不能把这组检查当作真实深度测量或检测准确率。

## 停止范围、测试和统计

没有运行静态 retarget、Transient Y、四种 gate 对比、30/50 个配对条件或 bootstrap。没有实现/验收 Cartesian Residual 注入、完整九事件视觉检测器、Runtime Skill 状态机及 OOD/Legacy drift 实验。`pilot_results.json`、`ablation.json`、`OOD_results.json` 明确标为 NOT_RUN，不能将这些缺项算成通过。

这是前置条件失败的 **NO-GO**，不是已完成配对实验但置信区间较宽的 INCONCLUSIVE。成功率提升、Event-Gated 相对 Phase-Gated 收益、检测 precision、false trigger rate 均未测得，结果字段保留 null。三个完整仿真成功记录不能替代配对统计。

最终回归：**169 passed，1 deselected，27 subtests passed**。其中 152 项旧测试和 17 项新测试通过。排除的旧测试会导入 `RIGHT_ARM_CONTROL_ROOT`，违反任务书明确限制；因此不声称“旧测试全部通过”。旧单测会检查纯历史 plan 逻辑；本轮实验 runtime 的 Legacy Skill 执行为 0。

本轮执行：2 次完整 native generator、1 次完整 frozen replay、1 次 grasp prefix、1 次 prefix contact replay。没有工程子 Agent。Runtime LLM、硬件执行、新的 Twin 参数/几何修改、Legacy Skill 执行均为 0。继续使用原有 rigid-collection 数值配置，不将其说成真实标定参数。原 generator 记录的 113 项源码/配置哈希复核全部一致。

源码与执行版本见 `source_manifest.json`；任务书、冻结设计和 checkpoint 已更新。真实/仿真原始视频、XML、资产、原始日志均留在原目录或 DATA_ROOT；Git 只保存新增代码、经审阅摘要、相对路径和哈希。

后续若另立任务，应先验证 RGB 中的目标身份和遮挡期间的可观测性，例如有证据的 RGB 时序对应或 housing-to-tip 关系；不能把 simulator pose 投影当作 Student 输入。当前批次没有任何待自动重试的动作。

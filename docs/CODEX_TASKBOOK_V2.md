# Codex 实施任务书 V2：统一主 Agent—快慢子系统—策略执行与可验证自进化

**目标仓库**：https://github.com/Adeline123456789/digital-twin （main，以执行时最新提交为准）  
**参考项目**：https://github.com/robosimstudio/robosimstudio  
**执行方式**：先在独立功能分支审计与实现，禁止默认接管真实机器人。  
**任务性质**：实际修改代码、增加可运行测试与客观报告，不是只给改造计划。

## A. 背景与不变量

1. 用户研究目标：从真实失败与模拟失败中快速定位第一处可观测异常，区分执行策略/感知/仿真模型错误；基于有证据的修复逐轮改善，保留经独立验证的技能，直到在指定任务分布上达到稳定表现；减少 Codex 推理时间与 Token。
2. 本仓库是 **Piper MuJoCo 旁路、离线研究层**，不是机器人完整仿真工程。先阅读 `AGENTS.md`、`protocol/robot_problem_solving.md`、`PROJECT_AUDIT.md`、`FINAL_RESEARCH_REPORT.md`、`ROUND2_REPORT.md`、`scripts/agent_loop.py`、`scripts/learn.py`、`scripts/diagnose_retention.py`、`scripts/compare_replays.py`、`scripts/alignment_skill.py`、`scripts/model_compare.py`、现有测试、`evolution/active.json`。
3. 当前实际事实：已建五类 JSON 记忆；已有限制的 LLM→MuJoCo→反馈闭环；存在 `compiled_alignment_v1` 适用范围内免 LLM 的验证技能；在历史仿真中抓取丢失 17.20 秒先于 18.30 秒 IK 报错；真机只有 RGB 与已有机器人状态/夹爪反馈，无六轴力传感器；不存在已经独立验证的真实物理校准与完整自动真机闭环；项目整体 NO-GO。不要改写这些事实。
4. 保留 AGENTS.md 数据路径、已有 three-camera 视频、Episode 格式、所有存储哈希/历史报告；不得把模拟 privileged object pose、接触力或来源不明的 depth 文件称为真机测量；不得把 headless 试跑伪装成完整成功采集。
5. **严禁在本任务中驱动真机**、跳过急停/限力/速度约束、放宽成功判定、修改控制器与真实安全配置、伪造校准、用固定约束/直接改物体位置造假成功。未得到新授权不得运行硬件动作。既有外部 `piper_simulation`、私有 XML/模型、真实视频与数据不得复制到 GitHub。
6. 不要假设单个 LLM 能判定真正物理根因；区分事实、推测和被证伪的解释。未知时输出 UNKNOWN，缺失传感数据必须 null。
7. **不以“每次尝试都必须成功”为目标**。只能保证未通过预注册回归门槛的候选不自动升级 active；失败和负经验要保留。这是可审计的非回退系统，不是保证无界单调成功率。

## A2. 必须采用的统一 Agent 体系（V2 核心修订）

**不要把快慢系统与主/子 Agent 写成两套架构。** 它们是同一个机器人运行时智能体的不同决策路径。必须明确区分：

1. **运行时 Supervisor（主 Agent / 逻辑调度器）**：维护目标、任务阶段、事件、预算和策略版本；可以主要由确定性代码实现，不要求每周期都调用 LLM。路由正常任务到快系统，未知或跨阶段的复杂问题到慢系统；持有唯一的当前执行状态（single-writer）。
2. **运行时 Fast Agents（快子 Agent/技能）**：Observation Agent（状态估计）、Event Agent（第一异常）、Skill Router（有效技能）、Policy Graph Executor（已验证任务动作）、Recovery Agent（有界恢复）。可以是规则/状态机/轻量模型，不是「每帧 LLM」。通过共享事件总线与慢系统通信。快系统**参与执行真机任务**，但只能通过经过授权的既有硬件接口发出允许的高层动作请求，绝不能替代独立安全控制器，也不得被自由生成代码直接驱动。
3. **运行时 Slow Agents（慢子 Agent）**：Task Planner、Diagnosis、World Model、Experiment Designer、Skill Evolution、Validation Reviewer。以 Codex/LLM 为工具按需调用，读取 Failure Capsule 和少量必要证据，产出有类型的候选计划/修复/证据。它们**不在电机伺服回路里**，需要先经过离线验证/批准才能影响真机。
4. **独立 Action/Hardware Safety Boundary**：策略图中的动作经 schema、范围/版本/授权检查后进入已有 Piper 机器人控制程序。已有安全联锁、急停、限力/速度/关节限制具有更高优先级，并不由 Agent 自行修改。当前任务授权范围为**纯离线**，不能进行真实运动。
5. **共享事实与事件总线**：真实传感、仿真 privileged、假设、策略版本、失败证据、候选修复/验证结论分类型存储；必须有来源和时间戳；Slow → Fast 只能通过 `CandidateProposal -> EvidenceGate -> ActivatedPolicy/Skill` 的单向审查路径进行升级。异常时 Fast → Slow 只发经过压缩的 Failure Capsule。
6. **开发时 Codex 主 Agent 与子 Agent**：只是用于**开发**上述运行时组件的一种工程协作编排，不等同于运行时的 Supervisor/Task Planner；开发子 Agent 并行与运行时快慢分层是**两个视角，不能互相替代**。

目标架构（接口关系，不宣称已经存在）：

```text
用户任务 / 任务目标
        ↓
运行时 Supervisor（主 Agent：状态+路由+预算）
        ├── Fast Agents（观测/异常/技能路由/策略图执行/有界恢复）
        │       ↓
        │   已验证任务动作 → 执行适配器 → 现有 Piper 安全控制 → 真机
        │       ↑         （当前仅有仿真/历史流式重放）
        └── Slow Agents（Codex：规划/诊断/世界模型/实验/验证）
                ↓
          Candidate → 离线仿真与回归 Gate → 版本化 Skill/Graph
                └──────────更新 Fast 可调用的已验证能力
共享：Observation/Event/Failure Capsule/Memory/Policy Version/Evidence
```

运行策略状态机至少支持：`IDLE -> PLAN -> EXECUTE_FAST -> VERIFY`；异常时 `EXECUTE_FAST -> HALT_OR_HOLD -> TRIAGE_FAST -> ESCALATE_SLOW -> OFFLINE_VALIDATE -> READY_FOR_APPROVAL`。`READY_FOR_APPROVAL` **绝非**未经授权直接恢复真机。只允许一个硬件动作命令所有者，不能让两个 Agent 并发给电机发指令。

**快慢时间含义**：快系统事件处理必须被测量，目标 p95 <20ms 是首版工程基准而非保证；真机 RGB 若只有30Hz，事件确认还受采样延迟限制；底层伺服与急停仍由已有实时/安全控制器承担。慢系统秒到分钟级，在机器人已安全停止、离线或非关键任务间隙运行。快系统不每帧重建 RSS/Newton 或修改全局摩擦。

## B. 先做实测基线，不要先重构

- 新分支 `feat/fast-slow-diagnosis`；保留默认 legacy 行为。记录当前 branch/HEAD、原仿真源码及其 dirty hashes，原 pytest/unittest 结果，保留当前环境限制。
- 通过已有 `scripts/agent_loop.py`、`scripts/benchmark_planner_context.py`、`scripts/benchmark_retention_diagnosis.py` 统计可复现的初始指标：模型调用次数、input/output tokens、完整推理/仿真/视频记录耗时、每个阶段首个观测到的失败事件、误报、未知率、保留成功 Episode/小时。不能运行的指标标 `NOT_RUN`，不能填零。
- 为历史报告、证据、版本建立只读索引；禁止将不同来源（original / retained_final）、不同 target、不同坐标系、不同 sensor 套件的运行无差别合并统计。
- 梳理当前 `scripts/model_compare.py:ask()` 每次启动 `codex exec --ephemeral` 的开销，把进程启动/基础工具上下文开销与任务历史上下文开销分开测量；当前有 ~20k 输入 Token 的实例，不要先断言全部来自聊天历史。

## C. 实施阶段 P0：一致的观测与事件数据合同（优先级最高）

新增最小包 `robot_agent/`，至少包含 `contracts.py`、`timeline.py`、`first_divergence.py`、`diagnosis.py`、`fast_loop.py`、`memory_router.py`、`gate.py`；保留现有 `scripts/*` 入口，允许适度调整实际目录以服从项目风格。

统一 `Observation` / `Event` / `FailureReport` schema，必须包含：

```
experiment_id, source_profile, native_source_hash,
world_model_version, policy_version, task_id, target_index,
phase, action_id, timestamp_mono, clock_source, time_alignment_quality,
frame_id, units, sensor_provenance, observability,
observed_value, predicted_value, residual, uncertainty,
kind, severity, status, evidence_refs, first_observed_frame,
first_observed_time, confirmation_time, hypothesis_ids,
root_cause_status
```

数据源分离：`real.rgb`、`real.proprio`、`real.gripper`、`sim.privileged_contact`、`sim.object_pose`。真实图像上的跟踪/姿态是估计值，必须有置信/可观测性信息，不得直接等同仿真 GT。时间轴以可复核时间戳对齐；不得假定历史 CSV 的命令时间等于硬件真实 CAN 发送时间。

实现由现有 Episode 数据直接读取的 `EventBuilder`；先基于已有 `scripts/diagnose_retention.py` 的事实检测器，产出 `GRASP_RETENTION_LOSS_OBSERVED`，再加入 `TARGET_FRAME_RESIDUAL`、`IK_UNREACHABLE`、`CONTACT_INACTIVE`（仿真专属）、`COLLISION_GAP`（仿真专属）、`SENSOR_UNAVAILABLE`、`TIMESTAMP_MISMATCH`、`INFRA_ERROR`。事件名称≠根因。

实现 `first_divergence`：先核查版本/时间/坐标/单位，再按 phase/action 顺序寻找最早**可观测**的显著偏差；支持滞回、连续样本、最小证据质量、观测间隔。输出 `first_observed_*` 与真实开始时刻 UNKNOWN 的区别。`ik_failure` 若晚于已证实 `grasp_retention_loss`，优先调查早期丢失，不应自动修复 IK。

P0 验收：不改变旧成功标签；历史 17.20s 夹持丢失和 18.30s IK 错误的顺序保持；时间戳错位、缺视频、缺传感、坐标单位错、模拟特权信息冒充真机都被拒绝或 UNKNOWN；含 6 条历史成功负对照的误报测试；新测试可在无机器人硬件条件下运行。

## D. P1：历史消息和记忆的严格检索

目标是不再给 LLM 输入完整历史聊天、全部 JSON 和整段视频，而是一次只拿一页 **Failure Capsule（失败胶囊）**。不要重新创建一套替代已有 `memory/{verified_skills,failure_patterns,negative_experiences,exploration_rules,digital_twin_knowledge}` 的目录。

建立轻量索引（优先 SQLite 标准库或者 JSONL；暂不要引入向量数据库、消息队列及 GPU 模型），提供 `retrieve_applicable(evidence, top_k<=5)`：
1. 硬过滤 source_profile/source_hash、task_family、target_index、frame、units、sensor_provenance、validation_status、适用初始范围；不可用的经验必须解释拒绝原因。
2. 将已证伪且受限的负经验作为 veto/警告，而不是把所有历史失败继续堆给模型。
3. 对适用候选再根据 failure event、phase、前因、证据质量和验证等级排序；禁止仅按字典序取前五条。
4. 缺少足够证据时返回 `UNKNOWN_NEEDS_OBSERVATION`，绝不默认为 `plug_not_seated`。
5. 生成约定上限的 Failure Capsule：任务及版本、3个最关键事件、第一处偏差、≤3个候选假设、≤5条检索经验引用、反例、下一项最小实验、禁止行为；只存原始文件的相对路径与 SHA256。
6. 用历史聊天摘要和 agent 日志时只提取可追溯的事实/决策/动作/结果/证据链接；无法核对的聊天内容标 `unverified_narrative`，不进入 verified skills。历史日志未知或不可访问就记录缺口，不推测存在。
7. 加入过期与冲突处理：源码哈希、场景、传感器变更则旧结论需要重新验证；一项新经验不能自动覆盖已有成功证据。

P1 验收：`VS002` 仅在原适用范围调用；retained_final / socket1 不得误用；已被撤销的真实深度声明不得进入检索；`NE011` 能阻止重复把 -10° 静态 IK 候选当作完整搬运修复；测量 prompt/token 与旧版对照，不能单靠更短 prompt 宣称任务可靠性提高。

## E. P2：真正的快慢双系统

### 快系统（运行时 Fast Agents，无 LLM，离线重放先实现）
- `FastLoop.update(observation) -> decision`，由运行时 Supervisor 管理、与 Policy Graph Executor/Recovery Agent 共用事件总线的常驻轻量 Python 流处理；不得每一帧启动 Codex、视频渲染、RSS、长仿真、模型部署或重建 XML。
- 先核查时钟/帧/单位/证据；再做有限状态机和滑动窗口检测。阶段如 `APPROACH`, `GRASP`, `LIFT`, `TRANSPORT`, `ALIGN`, `INSERT`, `RELEASE`, `VERIFY`，使用现有 phase label 作映射。
- 动态偏差采用鲁棒阈值 + 滞回 + 连续样本。可选接入 `river`（ADWIN 漂移）但先有无依赖 deterministic baseline。`ruptures` 只用于离线分段，不能说它是零延迟在线系统。
- 对于已通过 applicability 的经验，输出 `APPLY_VERIFIED_SIM_SKILL`；错误新颖/证据不足输出 `ESCALATE_SLOW`；数据缺失输出 `UNKNOWN`；危险/失控输出 `HALT_REQUEST` 给既有被授权的控制器安全接口（没有接口时只输出请求，不写硬件动作）。
- `FastLoop` 可以持续追踪实时观测和判断异常，但不得未经独立验证便自动修改全局真实摩擦、改动源代码或绕过控制器；当前仅作仿真/历史轨迹实时流重放。
- 单纯 30Hz 数据约 33ms 一个采样，3 帧持久性检测本身需要多个采样周期。按实际数据报告 p50/p95/p99 单事件处理延迟、从首个可观察异常到报告的检测延迟及漏报率，不得承诺真实硬实时。

### 慢系统（运行时 Slow Agents：Codex/LLM，仅非实时）
- 由同一个运行时 Supervisor 在快系统无匹配技能或需要复杂推理时触发。输入只有 Failure Capsule + 所需证据摘要 + 相关源码入口，不加载所有历史。
- 慢系统先决定是 `POLICY_ERROR`、`WORLD_MODEL_ERROR`、`PERCEPTION_OR_TIME_ERROR`、`CONTROL_OR_INTERFACE_ERROR`、`UNKNOWN`，每类只形成**假设**，不根据一个日志直接宣告根因。
- 输出最多3个有辨识能力的离线实验，按安全性、成本和信息价值排序。先测观测可用性/坐标/几何碰撞，再测接触保持，再测插入路径；不要因最后 IK 异常盲调位置。
- 每个补丁仅改变一类原因，另存 candidate/previous 版本和 evidence。可使用保留源代码的原生接口，在候选旁路复制/隔离配置下验证；不可修改独立成功检测器和安全限制。
- 运行预算必须受配置限制（调用次数、模拟次数、token、wall-clock），超额就停、报告 UNKNOWN，不得无限反思。

P2 验收：已有 VS002 适用时快路径零 LLM 调用；未知 socket1、缺真实深度、transport retention loss 等都能正确路由慢系统/UNKNOWN；快路径负对照误报可计数；断开 LLM、无 GPU 或 Codex 不可用时快系统仍能输出诊断或安全回退。

## F. P3：候选更新必须有安全门槛和版本回滚

沿用 `evolution/active.json` 与 JSONL 版本日志。新增 `CandidateProposal`、`EvidenceGate`、`PromotionDecision`，避免并行 Agent 隐式更新 active。

候选严格区分：`policy`、`sim_world`、`observation_adapter`、`diagnostic_rule`、`skill`。同一次对照只修改一个类别；需要跨类别联合修改时分两轮对照。每次候选必须有：失败机制、假设、预测变化、原始条件、预先锁定的 heldout 场景、反例、应保持的不变量、回滚指针。

检验次序：schema/单元测试 → 旧失败再现 → 独立保留轨迹回放 → 不同初始位置/姿态的接触保持与插入试验 → 完整 Episode 格式、原 native success/safety 检查 → 改进指标与成本。

设门槛：无碰撞/速度/关节极限放宽，无有效成功记录回退，无历史正例破坏，heldout 指标在统计不确定性内不显著恶化；要求与旧版本同源同配置对照，记录样本数和置信区间。未达门槛则 `REJECT_AND_ROLLBACK`；证据不足则 `DEFER_UNKNOWN`；不得写 REAL_VERIFIED。

新增防假成功测试：夹爪张开/脱离接触后，物体是否违反物理约束跟随运动；识别 MuJoCo contact 的 inactive vs active、geom gap、penetration、unexpected equality/weld、mocap 直改物体位置等（均仅模拟证据）。参考 RSS `robosimstudio/core/probes/contact.py` 和 RSS `examples/live/AGENTS.md`。RSS wipe_demo 使用了海绵 latch，不可直接搬到插头物理真实任务。

P3 验收：对不成功候选 active 版本字节级不变；测试 `-10°` tilt 的候选不可凭静态 IK 成功升级；在六条成功负对照上无新增误报；任何历史回放无法确认物理原因时不自动改摩擦；至少有一次实际的拒绝/回滚和一次可验证的适用范围内接受，且记录版本/成本。

## G. P4：RoboSimStudio 与 GaP 只做必要的增量接入

暂不替换当前 Piper MuJoCo、原控制器、真实记录格式。第一阶段优先复用 RSS 的设计与开源接口：
- `robosimstudio/core/probes/contact.py`：接触/穿透/法向力查询设计（在当前 MuJoCo native API 中实现等价观测）。
- `robosimstudio_real2sim_agent/checks.py` 与 `compare.py`：一次比较绑定场景源码哈希、图像对齐与 mask 位姿误差；真实 RGB 必须有可用的相机标定和置信。
- `examples/live/AGENTS.md`、`REFERENCE.md`：视觉校准先于物理调参，物理假成功检测与可复核报告。
- `examples/07_policies/README.md`：将策略做成可执行的 phase/skill graph，先简单 JSON graph 适配器，不一次性引入整个 GaP 与 Newton 工程。
- 如果确实需要新材料模型，在隔离的 sidecar sandbox 里单独评估 RSS，与原始 MuJoCo 同输入比较；给出安装/显卡/依赖兼容和具体性能后再决定合并。
- 参考 GaP `workflow.json` / `dag_trace.json`、postcondition checkpoint、trace diff 的概念；参考 ASPIRE 记录细粒度动作/感知/碰撞 trace 和验证后技能积累；不运行生成的任意 Python 到真机。

## H. Codex 主 Agent 与子 Agent 分工

开发时 Codex 主 Agent（不同于运行时 Supervisor）：审计已有实现、控制预算、决定任务顺序、处理子 Agent 结果冲突、亲自完成集成与 gate，**只允许主 Agent 合并 active 配置**。

最多同时2个子 Agent；独立只读工作可并行，写同一模块必须串行。创建 `.codex/agents/` 项目角色（当前 Codex 允许项目级 TOML；必须遵循官方最新 config），示例：
- `evidence_auditor`：只读检查历史 Episode、版本/时间/坐标/证据一致性，交付文件引用和基线。
- `diagnosis_engineer`：负责新的 event contract、first_divergence/fast loop，先生成方案和测试，主 Agent 指定时才写独立文件。
- `memory_engineer`：负责现有 memory 的适用检索与 evidence capsule，与 diagnosis_engineer 不同时修改 `learn.py`。
- `validation_reviewer`：只读审查物理假成功、测试隔离、回滚和安全边界，不具备 active 推送权限。

澄清：运行时 Fast Agents **是整个智能体体系中的子模块/子 Agent**，但并非 Codex Coding 子 Agent，通常没有 LLM。运行时 Slow Agents 才在需要高级推理时调用 Codex/LLM；开发时的 `.codex/agents` 仅是实现工程角色。三者必须共享同一事件/证据/版本约定，防止出现互不通信的两套系统。

## I. 交付文件和自动测试

实现后应新增/更新：
- `robot_agent/*.py` 包及 `scripts/triage_episode.py`；明确 `--offline`、`--local-only`、`--legacy`/`--fastslow` 的行为。
- `tests/test_observation_contract.py`、`test_first_divergence.py`、`test_memory_router.py`、`test_fast_loop.py`、`test_candidate_gate.py`、`test_fake_grasp.py`，用已有脱敏 fixture，不下载原始 MP4 到公共库。
- `docs/FAST_SLOW_ARCHITECTURE.md`、`docs/EVIDENCE_SCHEMA.md`、`experiments/<id>/RESULT.json` 和 README 快速使用说明。
- `.codex/agents/*.toml` 按现行官方文档进行角色定义，默认继承用户现有模型设置，不写死不可用模型。
- 在 `scripts/agent_loop.py` 添加 opt-in 模式，**默认 legacy**；把 `actual_planner_kind`、`model_invoked`、token、planner wall、simulation wall、diagnostic latency、evidence gate 统一计量。

最少测试情形：
1. 17.20s 夹持丢失先于18.30s IK错误；
2. 夹爪空握或隐式 weld 导致仿真假成功时拒绝模型；
3. 错误 time alignment、frame/units 和缺少 RGB 图像时进入 UNKNOWN，不伪造力学结论；
4. original/sock3 VS002 可匹配，retained/sock1 不能误匹配；
5. 负经验 NE011 不能被重复修复绕过；
6. 快系统无 LLM/无 GPU 仍能处理结构化事件；
7. 高成本新问题有预算，超出后可停止；
8. 候选无收益或回退时 `evolution/active.json` 不变；
9. 已有完整 Episode 成功判定不变，实验期间没有真实硬件调用或私有数据外传。

性能指标：准确率/误报/未知比例、first-divergence 延迟、模型调用次数/Token、仿真步数、回放和采集耗时、保留场景成功率与假成功率。报告原始 N 与置信区间；如果数据不足，写 `NOT_MEASURED` 而非计算虚假的改善。

## J. 运行轮数、分阶段预算、验收与停止规则（V2 新增强制量化）

**一轮（research/engineering round）**指：冻结一个问题、一个主要变更类别和预注册验收条件 → 实施补丁/候选 → 运行适用的单测/回放/仿真实验 → 保存定量结果/失败证据/回滚状态。它**不是**一次 LLM 回复、一次视频采样、一次 Episode 或一个训练 epoch。Codex 子 Agent 内部任务不计新研究轮次，必须归属当前主 Agent 管理的轮次。

### 第一批总预算：**14 轮工程/研究，新增完整 MuJoCo 执行上限 24 次，真机执行 0 次**

| 阶段 | 工程轮数 | 主要任务与每轮交付 | 新增完整 MuJoCo Episode 上限 | 历史记录回放/单测 |
|---|---:|---|---:|---|
| B：基线冻结 | **1** | 审计+固定分组、基线日志与 Token/延迟/结果 | **2** | 按可用历史原场景核对 |
| P0：观测/事件合同 | **2** | 轮1合同、时间坐标；轮2首个偏差/接触诊断 | **0** | 至少1条已知失败 + 6条成功对照，兼测缺失/错位 |
| P1：历史记忆与检索 | **2** | 轮1 Capsule/索引；轮2严格适用、撤销及负经验 | **0** | VS002/NE011/不同配置的单测 |
| P2：快慢运行时整合 | **3** | 轮1 Fast Agents；轮2 Supervisor+Slow 路由；轮3策略图执行/故障回退模拟 | **6** | 并行离线事件流重放与断 LLM 测试 |
| P3：候选改进/验证门槛 | **4** | 轮1物理假成功检查；轮2抓取保持；轮3候选与 heldout；轮4回归/回滚/学习记忆 | **12** | 原失败、独立保留成功、物理一致性回归 |
| P4：RSS/GaP 局部集成与最终对照 | **2** | 轮1只接最有价值观测接口；轮2 A/B 与文档报告 | **4** | 与 legacy 同源对照及独立 Reviewer 验证 |
| **合计** | **14** | 每轮独立结果与证据 | **24** | 不重复下载/复制原始视频 |

上限不是每个阶段一定要把物理试验用满：若单测未通过、独立测量缺失、候选回退或无意义，应立即 `DEFER_UNKNOWN`/`NO-GO`，余下预算**不得为了凑数盲跑**。若提前达到预注册目标可以提前结项。不同 profile/target 的成功数不得混合用于同一个稳定成功率。

**每一研究轮的硬上限**（不同于 24 次总体完整 Episode 上限）：最多 **1 个主错误假设类别 + 3 个候选假设 + 1 个主要变更类别 + 2 次慢系统 LLM 诊断调用 + 本阶段表内分配的模拟预算**。需要例外，先给出可审计理由和新预算，不能自动延长，也不能跨越原已封顶的历史 Stage2/3/4 研究轮数。

**轮次内验证**：始终先运行不涉及物理的 schema、单元测试、历史离线回放，再决定是否花费新 MuJoCo Episode。每轮必须输出 `round_id`, `hypothesis`, `changed_files`, `source_hash`, `baseline`, `candidate`, `heldout`, `test_count`, `sim_episode_count`, `llm_calls`, `input_tokens`, `output_tokens`, `wall_seconds`, `decision`, `unknowns`, `rollback_ref`。

**阶段验收阈值**：
- B：形成可重放的同源基线；如果环境缺失标 `NOT_RUN`，不可捏造数值。
- P0：已知 17.20s / 18.30s 事件顺序正确；6条历史成功负对照零新增误报；缺失/错位时 `UNKNOWN`。单一已知正例不宣称通用召回。
- P1：错误 profile/target/frame/units/sensor 被拒绝；已撤销 depth 不出现；VS002 在适用范围内直接执行；不支持范围 `UNKNOWN`。
- P2：离线流式执行能够给出 `CONTINUE`, `APPLY_VERIFIED_SIM_SKILL`, `ESCALATE_SLOW`, `HALT_REQUEST`, `UNKNOWN`；断开 Codex 仍运行；新事件不盲目重试；同源对比报告 p50/p95/p99 与 Token。
- P3：已知失败的早因不会被下游异常覆盖，任何假成功/隐藏 weld/越界均拒绝，必须演示至少1个 `REJECT_AND_ROLLBACK` 与1个适用范围内 `ACCEPT`（若真实无有效改进，则明确 `NO_ACCEPTED_CANDIDATE`，不造假）；旧成功和安全检查不得回退。
- P4：原系统功能与完整 Episode 结构兼容，同任务同预算 A/B 有独立结果；新代码和集成说明、版本回滚通过。

### 第二批：只有第一批通过后才开启**独立稳定性验证阶段 V**（不计入 14 轮）

先固定任务分布、原有安全边界、硬件可观测性和随机化种子，再采集至少 **36 条互相独立的 held-out 仿真 Episode**（建议3类位置/姿态子分布各12条）；每条都必须是完整有效数据，不能把重复渲染或同一轨迹回放视为独立尝试。给出成功率与 Wilson 95% 区间、假成功、接触/夹持、回归和单位成本。36 条只是筛选，不足以无条件声称“真实稳定成功”或“成功率超过95%”。

若用户将目标定义为「在指定且近似独立的仿真任务分布上成功率95%的双侧95% Wilson 置信下界」，则**零失败至少需要约73次独立试验**才能使下界刚超过95%；所以可在有证据后扩展至 80 条独立仿真 Episode，不满足就报告 NO-GO。该统计陈述不适用于相关性很高的重放、分布漂移或真机。

### 第三批：真机 R 阶段，**本任务不包含、不能自动执行**

只有用户另行明确授权、已有安全/硬件程序经过审计、现场具备急停及风险评估后，才设计新的执行阶段；初步预算可按 `R0` 只读/干跑（0运动）→ `R1` 至多3次受监督低风险动作 → `R2` 至多5次受监督完整任务 → 再按独立真实结果决定下一轮，不得把 3+5 视为已授权或稳定性统计。每个 R 阶段重新获授权并守独立停止门槛；安全问题立即终止。真机结果单独统计，不能由模拟成功替代。

## J2. 交付与进度可视化

按 B→P0→P1→P2→P3→P4 递进；每阶段必须提交**可运行代码/测试/对照数据/结论**才进入下一阶段。`experiments/round_budget.json` 记录预算上限、实际使用、拒绝和结余，`docs/FAST_SLOW_ARCHITECTURE.md` 同时画出**运行时主/快/慢/执行边界**与**开发时 Codex 主/子 Agent** 两幅不同层次的图，并写清它们的实现对应关系。

不得因完成进度而修改成功标准、合并不同仿真版本试验、丢掉失败 Episode、制造机器人成功或擅自延长已封顶研究。最终 `IMPLEMENTATION_REPORT.md` 明确报告 14 轮中各完成/阻塞的轮次、24 次 Episode 预算实际花费、慢 Agent 调用/Token、快系统延迟、回归、回滚及 REAL_NOT_TESTED。

若需要本地原有仿真资产但不存在，则继续可行的无依赖软件单元测试，标明 `NOT_RUN` 和缺失路径，在当前轮次内止损，不承诺后台继续工作。

## K. 开源资料（阅读具体实现，勿一口气引入全部依赖）

- RoboSimStudio：https://github.com/robosimstudio/robosimstudio （Apache-2.0；接触 probes、real2sim checks、policy examples）
- GaP：https://github.com/graph-robots/graph-as-policy （typed graph、checkpoints、trace-diff）
- ASPIRE：https://github.com/NVlabs/ASPIRE （execution trace → diagnosis → validated skill memory，注意生成代码沙箱）
- MuJoCo contact：https://mujoco.readthedocs.io/en/stable/computation/index.html#contact （margin/gap、active vs inactive contact）
- River：https://github.com/online-ml/river （可选流式漂移检测 ADWIN）
- ruptures：https://github.com/deepcharles/ruptures （离线变点检测，不当成硬实时）
- BehaviorTree.CPP：https://github.com/BehaviorTree/BehaviorTree.CPP （若之后正式 ROS2/C++ 执行图，可参考类型化非阻塞 actions）
- Codex 子 Agent：https://learn.chatgpt.com/docs/agent-configuration/subagents （角色的项目级 TOML 配置）

> 最终目标不是无限对话中的“看起来更聪明”，而是在真实证据边界内用更少的观察/推理/试验，找到最早异常，明确需要验证的假设，拒绝无效修复，并在独立测试中逐轮提升稳健成功率。

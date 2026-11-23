# Event-Driven Residual Adaptation：双臂仿真最小验证任务书 V1

## 0. 研究目标

项目：

`digital-twin`

核心研究问题：

能否利用冻结的仿真 Nominal Reference、部署可用的观测和 Interaction Event Divergence，在不修改 Digital Twin 的情况下，通过有界 Residual 恢复由于场景或交互变化造成的任务失败？

本阶段只验证：

1. Frozen Nominal 是否能够被正确冻结和重放。
2. Event Comparator 是否能正确区分五种事件比较结果。
3. Event-Gated Residual 是否比 Phase-Gated Residual 更有效。
4. UNKNOWN 是否能够阻止错误修正。
5. 旧 Skill 是否会被新实验错误调用。

本阶段不证明真实 Sim-to-Real 成功率提升，不训练 Residual Neural Network。

## 1. GitHub 分支与环境

仓库：

`https://github.com/Adeline123456789/digital-twin`

历史研究分支：

`research/right-arm-20261116`

推荐开发基线：

`feat/fast-slow-diagnosis`

基线 Commit：

`87bd88181836bed5c4723bc7a79a2586060d032f`

新建工作分支：

`research/event-residual-dual-20261123`

命令：

```bash
git fetch origin
git switch -c research/event-residual-dual-20261123 \
  origin/feat/fast-slow-diagnosis
```

不得修改原来的两个分支。

环境变量：

```bash
export PROJECT_REMOTE=https://github.com/Adeline123456789/digital-twin

export RESEARCH_BRANCH=research/right-arm-20261116
export BASE_BRANCH=feat/fast-slow-diagnosis
export WORK_BRANCH=research/event-residual-dual-20261123

export PIPER_SIM_ROOT=/media/midea/F496AC8096AC4548/piper_simulation

export PIPER_DATA_ROOT=/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug
```

保持现有 MuJoCo 环境和原有数据路径，不复制仿真模型、XML、资产和原始录像到 Git。

## 2. 实验权限

本阶段严格限制：

```yaml
experiment_mode: event_residual_mvp

robot_arms: dual

hardware_execution: deny

real_data_read: deny

real_data_training: deny

llm_execution: deny

twin_mutation: deny

legacy_skill_execution: deny

legacy_memory_access: warning_only

nominal_policy: frozen_reference

residual_policy: deterministic_v0

event_observer: rule_based_v0

recovery_router: contract_guarded_v0
```

MVP 不得读取：

- REAL_FAILED
- REAL_SUCCEEDED
- REFERENCE_DATA
- RIGHT_ARM_CONTROL_ROOT

不调用真实机器人 API、CAN、使能、运动和回放。

第一阶段只使用：

- 已有双臂 MuJoCo 场景；
- SIM_SUCCEEDED 中经过验证的成功数据；
- 新生成的仿真 Episode；
- MuJoCo 渲染 RGB；
- 仿真关节和夹爪观测；
- 仅供 Teacher / Evaluator 使用的仿真特权信息。

## 3. Stage 0：现有系统审计

开始写新代码之前，检查：

### 3.1 仿真任务

重点阅读：

- `scripts/collect_current_scene.py`
- `scripts/benchmark_collection.py`
- `scripts/learn.py`
- `scripts/collect_inline.py`
- `scripts/episode_format.py`
- `$PIPER_SIM_ROOT/scripts/try_plug_pick_sim.py`

确认：

1. 双臂 MuJoCo 模型加载方式。
2. 原生任务实际使用哪些机械臂。
3. 左右臂 action 的实际维度、单位和索引。
4. 原生动作控制接口。
5. 插头、插座和夹爪的名称。
6. 原生 phase 顺序。
7. 原生插入控制是否包含内部重新规划。
8. 是否可以在控制循环中实时注入 Cartesian residual。

特别注意：

目前历史报告表明原生任务主要执行右臂，左臂没有参与插头支撑。

因此本阶段定义为：

双臂机器人模型 + 现有右臂插头任务。

不得声称已经实现双臂协同。

如果要测试真正的双臂协同，先单独记录为后续研究，不要擅自增加左臂控制逻辑。

### 3.2 原有 Fast/Slow

检查：

- `robot_agent/contracts.py`
- `robot_agent/diagnosis.py`
- `robot_agent/fast_loop.py`
- `robot_agent/first_divergence.py`
- `robot_agent/supervisor.py`
- `robot_agent/memory_router.py`
- `robot_agent/local_slow.py`
- `robot_agent/gate.py`
- `docs/FAST_SLOW_ARCHITECTURE.md`
- `docs/EVIDENCE_SCHEMA.md`

保留现有 provenance、clock、evidence 和 safety 约束。

不要直接让旧 `agent_loop.py --arm D` 接管新实验。

## 4. Stage 1：生成并冻结 Nominal Reference

使用当前原生任务作为 Nominal Generator。

首次基线：

```bash
"$PIPER_SIM_ROOT/.venv-mujoco-hardware/bin/python" \
scripts/collect_current_scene.py \
--robot-arms dual \
--offset-mm 0 \
--shift-mm 0 \
--local-only
```

要求：

- 原生任务成功；
- 原有成功检测器通过；
- 完整 Episode 验证通过；
- 保存原始 source/config/adapter hashes。

从成功轨迹中导出 Frozen Reference。

建议新增：

`robot_agent/nominal_reference.py`

导出内容：

- timestep；
- phase；
- nominal action；
- nominal joint target；
- action semantics；
- reference episode ID；
- source hash；
- task config hash；
- policy hash；
- initial state hash。

原始轨迹仍保留在 DATA_ROOT。

Git 只保存 manifest、摘要和 SHA-256。

### Frozen Reference 的执行约束

后续每次 rollout：

- 使用同一 reference；
- 不重新调用 Planner；
- 不读取当前 privileged goal pose 来修改 reference；
- 不允许根据 perturbation 重新生成动作；
- 保留低层原生跟踪控制器。

建立 `ReferencePlayer`。

如果原生控制器无法直接播放冻结参考，则增加最小的 replay adapter。

不得用简单的离线数组比较冒充真实执行冻结参考。

### Privileged Access Audit

建立：

`PlannerSpy`

`PrivilegedAccessAudit`

在 Reference 冻结后记录：

- Planner 调用次数；
- policy 的 privileged state 读取次数；
- goal/object/contact 访问次数；
- reference hash；
- nominal action hash。

策略不得持有完整 MuJoCo `MjData` 对象。

物理引擎正常读取内部状态不属于策略泄漏。

必须通过接口隔离策略输入。

## 5. Stage 2：实现 Event Contract

新增：

`robot_agent/interaction_events.py`

定义：

`EventSpec`

`ObservedEvent`

`EventComparison`

`InteractionEventTrace`

每个事件包含：

- event_id；
- event_name；
- phase；
- expected_presence；
- earliest_time；
- deadline；
- prerequisites；
- observation_requirements；
- status；
- confidence；
- timestamp；
- evidence_ref。

### 5.1 Event Observation

允许：

`TRUE / FALSE / UNKNOWN`

### 5.2 Event Comparison

实现：

`MATCHED / PENDING / CONTRADICTED / UNRESOLVED / UNEXPECTED`

具体规则：

1. 未到事件窗口：PENDING。
2. 预期事件在有效窗口内被确认：MATCHED。
3. 预期事件截止时间已过，可靠观测确认没有发生：CONTRADICTED。
4. 截止时间已过，但必要观测缺失或不可靠：UNRESOLVED。
5. 出现不符合预期的事件或事件顺序：UNEXPECTED。

对于 expected absence：

- 没有出现事件且观测可靠：MATCHED。
- 出现了禁止或未预期的事件：UNEXPECTED。
- 无法确定是否出现：UNRESOLVED。

必须处理时钟不连续、观测丢失和置信度不足。

### 5.3 Event Sequence

第一版实现：

- APPROACH_REACHED
- GRASP_ESTABLISHED
- LIFT_CONFIRMED
- RETENTION_OK
- PREINSERT_REACHED
- ALIGNMENT_OK
- INSERTION_PROGRESS
- SEATED
- RELEASE_STABLE

其中真实可部署事件不能直接使用 MuJoCo 特权信息。

Sim privileged events 只作为独立 Ground Truth。

### 5.4 三值逻辑

实现 Strong Kleene 三值逻辑：

AND、OR、NOT。

例：

`TRUE AND UNKNOWN = UNKNOWN`

`FALSE AND UNKNOWN = FALSE`

`TRUE OR UNKNOWN = TRUE`

`FALSE OR UNKNOWN = UNKNOWN`

`NOT UNKNOWN = UNKNOWN`

测试事件 predicate 的 UNKNOWN 传播。

### 5.5 关键安全规则

UNRESOLVED：

- 不触发 Residual；
- 不触发 Structural Update；
- 不调用 LLM；
- 不将它视为 MATCHED。

关键事件无法观测时，独立安全监督器决定 HOLD 或 SAFE_STOP。

## 6. Stage 3：Observation Adapter

新增：

`robot_agent/observation_adapter.py`

`robot_agent/interaction_observer.py`

第一版只使用部署可获得的观测：

- joint position；
- joint velocity；
- command-feedback residual；
- gripper width；
- rendered RGB；
- plug image features；
- socket image features；
- relative image residual；
- visual confidence；
- current phase；
- nominal action。

禁止读取：

- exact plug 6DoF；
- exact socket 6DoF；
- simulator contact pairs；
- simulator insertion depth；
- exact collision penetration；
- simulator task success。

视觉第一版优先使用可解释的 OpenCV 特征提取。

至少输出：

- plug tip pixel；
- socket center/keypoints；
- relative image residual；
- confidence；
- visibility / occlusion status。

不能可靠检测时输出 UNKNOWN。

不要把仿真 privileged pose 投影成 Student 输入。

## 7. Stage 4：Static Retarget 实验 A

目的：

验证静态几何变化是否可以通过目标重新定位解决。

扰动：

Socket X shift。

预声明测试：

- -2 mm；
- -1 mm；
- 0 mm；
- +1 mm；
- +2 mm。

先做 pilot。

如果 nominal 全部成功或全部失败，重新选择有区分度的测试范围，并在正式测试前冻结。

比较：

A0：Frozen Nominal Only。

A1：Frozen Nominal + Deployable Retarget V0。

A2：VS002 Privileged Oracle（可选）。

A2 只有在历史 Skill 的 source hash、双臂结构、socket、frame、units、contact 和初始状态条件全部通过时才允许执行。

VS002 使用仿真特权残差时必须明确标记为 Oracle，不得称为 Real-deployable Retarget。

所有新实验都禁止自动调用旧 Skill。

指标：

- success rate；
- target residual；
- number of replanning calls；
- residual magnitude；
- Twin modification count。

这一实验不作为 Event-Gating 创新证据。

## 8. Stage 5：Transient Perturbation 实验 B

这是核心实验。

目标：

验证 Event-Gated Residual 是否比 Phase-Gated Residual 更有效。

### 8.1 扰动设计

在：

`PREINSERT_REACHED`

之后，给 Socket Y 方向施加短暂位移。

扰动必须同时改变：

- 实际物理碰撞几何；
- 相机渲染结果。

不能只改变控制器内部的目标坐标。

优先检查原生 Socket Body 是否支持 mocap 或可控运动。

如果不支持，设计一个最小的运行时运动学扰动接口。

不修改摩擦、质量、碰撞参数和成功阈值。

扰动时间曲线：

- PREINSERT 前：0；
- 预定扰动时刻：施加 Y 偏移；
- 固定时间窗口内保持偏移；
- 随后恢复。

扰动持续时间固定，起始事件固定。

正式实验只改变预声明的扰动幅度。

使用模型实际几何间隙确定 small / medium / large，不得随意猜毫米数。

### 8.2 Residual Controller

新增：

`robot_agent/residual_controller.py`

第一版使用确定性 Visual Servoing。

根据 RGB 相对特征误差计算有界修正。

动作必须通过实际控制接口注入。

记录：

- nominal_action；
- residual_action；
- final_action；
- gate；
- correction norm；
- correction duration。

Residual 只能修改右臂插入相关动作；左臂保留原生参考动作。

如果没有真实可用的动作注入点，停止并报告接口阻塞。

### 8.3 四组对照

A0：Nominal Only。

`gate = 0`

A1：Ungated Residual。

整个任务期间允许 Residual。

A2：Phase-Gated Residual。

仅当 phase 为 align / insert 时允许 Residual。

A3：Event-Gated Residual。

只有已确认的可恢复交互事件偏差才能开启 Residual。

三种 Residual 实验必须使用同一个 residual control law 和 action bounds。

唯一改变的是 gate。

### 8.4 Event Gate

允许触发的条件：

- observation 可用；
- confidence 达标；
- phase 在允许范围；
- 事件属于已验证的恢复类型；
- action 在允许范围。

例如：

`ALIGNMENT_ERROR` 被可靠检测为 UNEXPECTED，并且该事件在已验证 recovery map 中。

或：

`ALIGNMENT_OK` 在截止时间后被 CONTRADICTED。

恢复后，连续若干采样确认 MATCHED，关闭 Residual。

UNRESOLVED 不触发 Residual。

未知 UNEXPECTED 不自动执行动作。

## 9. Stage 6：OOD / UNKNOWN / Legacy Contract 实验 C

测试三种情况：

### C1：Observation Missing

人为遮挡关键视觉特征。

预期：

`UNKNOWN → UNRESOLVED → NO_ACTION / HOLD`

### C2：OOD Perturbation

扰动幅度超过预声明训练/验证范围。

预期：

`UNKNOWN` 或 `STRUCTURAL_CANDIDATE`

不能直接修改 Twin。

不得读取 simulator_success 决定路由。

### C3：Legacy Skill Drift

构造与 VS002 不匹配的 source / embodiment / frame / observation contract。

预期：

`SUSPENDED / NOT_APPLICABLE`

旧 Skill 不得执行。

### Runtime Contract

新增：

`robot_agent/skill_contract.py`

`robot_agent/skill_registry.py`

至少包含：

- task；
- embodiment；
- source hash；
- world model version；
- policy version；
- observation contract；
- required events；
- preconditions；
- action bounds；
- postconditions；
- validation scope；
- drift policy。

状态：

CANDIDATE、ACTIVE、SUSPENDED、RETRACTED、RETIRED。

一次失败不能自动全局删除 Skill。

先暂停当前适用范围，再经过证据审核决定是否撤回。

## 10. Recovery Router

新增：

`robot_agent/recovery_router.py`

输出：

- NO_ACTION
- RETARGET
- RESIDUAL
- STRUCTURAL_CANDIDATE
- UNKNOWN

规则：

MATCHED / PENDING → NO_ACTION。

可验证的静态几何偏移 → RETARGET。

已知、可观测、处于恢复范围内的交互偏差 → RESIDUAL。

UNRESOLVED → NO_ACTION，关键安全事件可 HOLD / STOP。

未知事件 → UNKNOWN。

重复且有证据支持的新型结构性问题 → STRUCTURAL_CANDIDATE。

STRUCTURAL_CANDIDATE 不能直接修改 Twin。

必须经过 Slow Agent 离线分析和 Evidence Gate。

## 11. Fast/Slow 集成

复用：

- EventBuilder；
- FastLoop；
- Supervisor；
- Failure Capsule；
- MemoryRouter；
- EvidenceGate。

但新 MVP 必须使用独立运行入口：

`scripts/run_event_residual_mvp.py`

不通过旧 `agent_loop.py --arm D` 启动。

Fast：

- 读取 deployable observation；
- 判断 event；
- 比较 reference；
- 执行有界 gate；
- 不调用 LLM。

Slow：

- 仅在离线诊断阶段使用；
- 读取 failure capsule；
- 生成有限假设和离线实验建议；
- 不直接控制机器人；
- 不自动修改 Twin。

研发子 Agent 由主 Codex 管理预算和验收。

保持一次只运行一个受限工程子任务的原则。

## 12. 自动化测试

至少实现：

- `tests/test_event_three_valued_logic.py`
- `tests/test_event_comparator.py`
- `tests/test_event_sequence.py`
- `tests/test_pending_deadline.py`
- `tests/test_unresolved_no_action.py`
- `tests/test_unexpected_event.py`
- `tests/test_nominal_reference.py`
- `tests/test_nominal_action_invariance.py`
- `tests/test_privileged_access_audit.py`
- `tests/test_residual_bounds.py`
- `tests/test_event_gate.py`
- `tests/test_phase_gate.py`
- `tests/test_recovery_router.py`
- `tests/test_skill_runtime_contract.py`
- `tests/test_skill_suspension.py`
- `tests/test_legacy_skill_isolation.py`
- `tests/test_observation_no_privileged_leakage.py`

旧测试必须全部通过。

## 13. 实验预算与统计

第一轮 Pilot：

- 静态 X：预声明偏移网格；
- Transient Y：三个预声明幅度；
- 每种条件至少进行少量配对 pilot。

正式确认：

- 至少 30 个配对测试条件；
- 每组使用相同 initial state / random seed；
- 四种策略比较；
- 保留独立 held-out seeds；
- 如统计置信区间仍然过宽，增加到 50 个以上配对条件。

正式评估不得继续调节扰动直到得到想要的结果。

### 记录指标

1. task success rate；
2. nominal-only failure rate；
3. Event-Gated recovery rate；
4. Phase-Gated recovery rate；
5. residual action norm；
6. unnecessary residual activation；
7. false trigger rate；
8. event detection precision；
9. event detection delay；
10. unresolved rate；
11. privileged policy reads；
12. planner calls；
13. LLM calls；
14. legacy skill invocations；
15. Twin modifications。

## 14. MVP GO / NO-GO

同时满足：

1. 双臂模型加载、原生任务和完整 Episode 验证通过。
2. Frozen Reference 在无扰动条件下可重放成功。
3. Nominal 冻结后无隐式重新规划和 privileged policy reads。
4. Transient 扰动确实使 Nominal 在一部分预声明条件下失败。
5. Event Comparator 五种状态正确。
6. UNKNOWN 不被误报为 CONTRADICTED。
7. UNRESOLVED 不触发 Residual / Structural Update。
8. Event-Gated 相比 Phase-Gated 有可测收益：
   - 成功率有明确提升；或者
   - 成功率基本相当，但无效修正量至少减少 25%。
9. 事件检测准确率达到预声明目标，建议不低于 90%。
10. 无扰动条件的 false trigger rate 不超过 5%。
11. Privileged policy reads = 0。
12. LLM calls = 0。
13. Legacy Skill invocations = 0。
14. Twin modifications = 0。
15. Hardware executions = 0。
16. 旧测试无 regression。

对配对结果进行 bootstrap 置信区间分析。若置信区间无法支持结论，标记为 INCONCLUSIVE，而不是 GO。

### NO-GO 时

停止。

不允许自动：

- 训练 PPO；
- 扩大 Agent 数量；
- 增加 Memory；
- 修改摩擦系数；
- 重建 Twin；
- 修改成功阈值。

报告具体失败属于哪一种：

- nominal interface failure；
- observation contract failure；
- event detection failure；
- residual action interface failure；
- event gate failure；
- perturbation design failure。

## 15. GitHub 发布

每个阶段结束后：

- 运行测试；
- 保存实验设计；
- 保存结果；
- 保存代码版本和哈希；
- 更新研究 checkpoint；
- Commit；
- Push。

目标目录：

```text
docs/
    EVENT_RESIDUAL_MVP_TASKBOOK.md
    EVENT_RESIDUAL_EXPANSION_PLAN.md

robot_agent/
    nominal_reference.py
    observation_adapter.py
    interaction_observer.py
    interaction_events.py
    event_comparator.py
    residual_controller.py
    recovery_router.py
    skill_contract.py
    skill_registry.py

scripts/
    run_event_residual_mvp.py

tests/
    test_event_three_valued_logic.py
    test_event_comparator.py
    test_nominal_action_invariance.py
    test_privileged_access_audit.py
    test_event_gate.py
    test_recovery_router.py
    test_skill_runtime_contract.py

experiments/
    event_residual_mvp/
        design.json
        pilot_results.json
        results.json
        ablation.json
        OOD_results.json
        MVP_REPORT.md
        source_manifest.json
```

正式实验前后的完整代码和精简报告必须推送到：

`research/event-residual-dual-20261123`

禁止上传：

- 原始真实录像；
- 原始仿真录像；
- 私有 MuJoCo XML；
- 仿真资产；
- 机器人密钥；
- 私有控制程序；
- 未经审核的本地日志。

原始数据留在 DATA_ROOT，GitHub 只保存相对路径、哈希和脱敏统计结果。

最终必须确认：

```bash
git status
git log -1
git push -u origin research/event-residual-dual-20261123
git ls-remote origin refs/heads/research/event-residual-dual-20261123
```

远端 HEAD 必须和本地最终 Commit 一致。

最终报告必须明确：

GO / NO-GO / INCONCLUSIVE。

不得把 Simulation-only 结果描述成 Real-verified。

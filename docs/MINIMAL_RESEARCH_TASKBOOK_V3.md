# MINIMAL RESEARCH TASKBOOK V3
# Full-Episode Post-Grasp Residual Recovery

## 0. 固定研究假设

研究假设 H1：

在完整的插头抓取与插入任务中，如果插头在抓取完成后相对于夹爪发生位姿偏移，那么只在仿真中训练的 Proprioceptive History Residual，能否利用真实可部署的观测，在 Frozen Nominal 的基础上修正后续动作，恢复任务成功？

本实验是完整任务的后抓取恢复验证，不代表已经完成所有阶段的 Residual 学习。

本阶段覆盖：

- GRASP_ESTABLISHED 之后的状态偏差；
- LIFT；
- TRANSPORT；
- ALIGN；
- INSERT；
- RELEASE；
- RETREAT。

暂不训练 PRE-GRASP 和 GRASP 阶段的 Residual。

这只是为了控制第一轮训练范围，不是最终研究范围的限制。

## 1. GitHub 和开发环境

Repository:

https://github.com/Adeline123456789/digital-twin

Base branch:

`research/event-residual-dual-v2-20261123`

Base commit:

`26649c5446b972ed366969cbff5a460b3b6ae697`

新分支：

`research/full-task-residual-20261124`

```bash
git fetch origin

git switch -c research/full-task-residual-20261124 \
origin/research/event-residual-dual-v2-20261123
```

保留所有历史分支：

- research/right-arm-20261116
- research/event-residual-dual-20261123
- research/event-residual-dual-v2-20261123

以后整个 Full-Task Residual 研究在新分支中逐阶段推进，不再为每个小实验创建大量分支。

环境变量使用：

```bash
export PROJECT_REMOTE=https://github.com/Adeline123456789/digital-twin

export SIM_ROOT=/media/midea/F496AC8096AC4548/piper_simulation

export RIGHT_ARM_CONTROL_ROOT=/media/midea/F496AC8096AC4548/act_td3/piper_use_right

export RIGHT_ARM_RUNBOOK=/media/midea/F496AC8096AC4548/act_td3/piper_use_right/piper记事本_单右臂.txt

export DATA_ROOT=/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug

export SIM_SUCCEEDED=$DATA_ROOT/complete_sim/succeed

export SIM_FAILED=$DATA_ROOT/complete_sim/failed
```

REAL_FAILED、REAL_SUCCEEDED、REFERENCE_DATA 只保留路径配置，本轮不读取。

## 2. 权限约束

```yaml
experiment_mode: full_task_post_grasp_residual

robot_scene: dual_arm

active_task_arm: right

left_arm: frozen_nominal

hardware_execution: deny

real_data_read: deny

real_training: deny

llm_runtime: deny

legacy_skill_execution: deny

automatic_twin_mutation: deny

nominal_update: deny
```

不允许：

- 导入真实机器人驱动；
- 调用 CAN；
- 使能真实机械臂；
- 修改真实控制程序；
- 修改 MuJoCo friction；
- 修改 contact solver；
- 修改任务成功阈值；
- 使用新的仿真特权信息重新生成 Nominal。

## 3. Stage A：复现并确认 V2 Residual

先复现现有 V2。

保留原始：

- Frozen Reference；
- B3 Structured checkpoint；
- 训练配置；
- held-out 32 条条件；
- Nominal 20 条条件。

随后增加三个独立训练 seed：

51001、51002、51003。

必要时增加 51004、51005。

重新训练 Proprioception-only History PPO。

使用：

- 15 帧历史；
- 右臂 joint position；
- joint velocity；
- nominal action；
- controller feedback residual；
- gripper width；
- phase。

不使用 RGB 输入。

每个 seed 16,384 steps，固定训练预算。

比较：

B0：Frozen Nominal。

B1：Deterministic Local Search。

B2：History PPO Structured Reward。

正式测试使用新的 64 个 held-out 条件，并保留 20 个 nominal regression 条件。

使用配对测试和按训练 seed / test condition 分层的 bootstrap。

### GO

- B2 平均成功率相对 B0 提升至少 15 个百分点；
- 95% CI 下界大于 0；
- 至少 4/5 个 seed 有正向提升；
- Nominal regression 不超过 5 个百分点；
- 不发生动作越界。

同时报告 B2 与 B1 的差异。

如果 PPO 不优于确定性搜索，不得声称 PPO 优于传统恢复方法。

本阶段 NO-GO 时停止，不进入 Full-Task PPO。

## 4. Stage B：完整 Nominal Reference 审计

复用已有完整成功 Episode。

历史报告记载原始完整 Episode 为 1133 帧。

必须实际检查：

- reference.jsonl 的真实长度；
- phase 顺序；
- 每帧 action；
- 双臂 action 维度；
- gripper target；
- 时间戳；
- 原始成功检测器；
- 完整 Episode validator。

不能仅因为存在一个 1133 帧的录像，就认为已经有 1133 帧可执行动作。

如果现有 Frozen Reference 不完整，从已有成功 Episode 中导出完整参考。

保存：

`full_task_reference.jsonl`

`full_task_reference_manifest.json`

必须包括：

- source episode；
- source hash；
- action semantics；
- units；
- phase；
- timestep；
- left/right arm targets；
- gripper target；
- success provenance。

完整 Nominal 从 APPROACH 到 RETREAT。

重放时：

- 不重新调用 IK Planner；
- 不重新生成 trajectory；
- 不读取扰动后的 privileged object pose；
- 保留原低层关节跟踪控制器；
- 左臂执行原始参考。

必须通过无扰动完整任务的重放和完整 Episode 验证。

## 5. Stage C：Post-Grasp Observation Audit

目的：

在训练之前确认插头相对于夹爪的位姿变化能否被部署可用观测识别。

只测试两种观测：

P0：Proprioception History。

- joint position；
- joint velocity；
- nominal action；
- command-feedback residual；
- gripper width；
- phase。

P1：P0 + Right Wrist RGB。

RGB 只使用：

- 黄色插头外壳特征；
- 夹爪左右手指特征；
- 插头相对于夹爪的二维相对位置；
- 可见性和置信度。

不要求同时看到 Top Pin 和 Top Hole。

不使用 overhead camera 冒充真实 right-wrist camera。

必须核实仿真相机与真实 `cam_right_wrist_rgb.mp4` 的视角对应关系。

外部摄像头仅作为后续目标定位输入，本阶段固定插座位置，不验证目标定位算法。

### Observability Experiment

生成配对 Episode：

A：Nominal grasp。

B：GRASP_ESTABLISHED 后插头发生已知相对位姿偏移。

比较 P0/P1 的历史输入。

评估：

- 是否能够区分有无偏移；
- 能否判断偏移方向；
- 是否能估计偏移趋势；
- 延迟多少帧能够识别；
- 是否存在不可观测情况。

按完整 Episode / seed 划分训练和测试，不能把相邻视频帧当成独立测试样本。

建议初始验收：

- 扰动检测 AUROC ≥ 0.75；
- 四方向分类准确率 ≥ 0.60；
- 测试必须是 held-out Episode。

这些是本项目的工程验收阈值，不是通用学术标准。

P0 通过则优先使用 P0。

P0 不通过、P1 通过，则允许使用 P1。

P0 和 P1 都不通过：

停止 Residual Training，报告 OBSERVATION_UNIDENTIFIABLE。

不允许临时添加传感器、扩大网络或者调用 Agent 猜测状态。

## 6. Stage D：Full-Episode Post-Grasp Perturbation

这是本轮核心实验。

### D1. 扰动发生位置

完整 Nominal 先正常执行：

APPROACH → GRASP → GRASP_ESTABLISHED。

随后在 LIFT 开始之前施加扰动。

扰动对象：

插头相对于右夹爪的位姿。

第一轮只研究 XY 平移，不同时训练 yaw 和 gripper dynamics。

扰动在夹爪坐标系中定义。

建议 pilot：

- 0.25 mm；
- 0.5 mm；
- 0.75 mm；
- 1.0 mm；
- 1.25 mm；
- 1.5 mm。

结合模型真实碰撞间隙，确定有效范围。

正式训练和测试范围必须在 held-out 运行之前冻结。

这些数值是仿真压力测试范围，不代表已经测得真实世界的滑移分布。

### D2. 区分两种扰动

第一种：

CONTROLLED_POST_GRASP_STATE_PERTURBATION。

在 GRASP_ESTABLISHED 后改变插头相对夹爪状态。

这是反事实状态扰动，用来测试可恢复性。

不能称为物理摩擦滑移。

第二种：

CONTACT_MEDIATED_DRIFT。

使用 MuJoCo 外部扰动力或现有物理接触机制，尝试让插头在夹爪中自然发生相对运动。

不修改摩擦系数。

如果当前模型无法产生可重复的相对运动，报告 PHYSICAL_SLIP_NOT_REPRODUCED。

不得把第一种扰动冒充第二种。

### D3. 观测和动作

Student 输入：

15 帧历史：

- right joint q；
- right joint dq；
- nominal action；
- command-feedback residual；
- gripper width；
- phase；
- 可选的右腕 RGB 相对特征。

禁止：

- exact plug pose；
- exact socket pose；
- exact contact pair；
- insertion depth；
- penetration；
- simulator success。

第一版输出：

`Δx, Δy`

作用于右臂末端 XY 参考位置。

使用完整 Reference 对应的逐帧 Jacobian。

左臂动作和夹爪动作保持 Nominal。

Residual 从 GRASP_ESTABLISHED 之后开始启用，在 LIFT、TRANSPORT、ALIGN、INSERT 阶段参与。

RELEASE 和 RETREAT 按预先定义的平滑退出规则执行。

使用 V2 的初始安全边界：

- 每轴 ±2 mm；
- 速度 ≤3 mm/s；
- 加速度 ≤30 mm/s²；
- joint correction ≤0.03 rad。

本轮不得为了提高成功率擅自扩大边界。

## 7. Stage E：Full-Task Residual Training

新增独立环境：

`robot_agent/full_task_residual_env.py`

不要修改 V2 原始局部环境。

训练新模型：

`FullTaskPostGraspResidual`

保留 V2 B3 作为历史基线，不覆盖旧 checkpoint。

网络：

- 15 帧 History Encoder；
- Phase Embedding；
- 小型 MLP；
- XY Residual Head。

Phase：

- LIFT；
- TRANSPORT；
- ALIGN；
- INSERT；
- RELEASE；
- RETREAT。

Student 只读取部署可用输入。

Privileged Information 仅用于：

- Reward；
- Teacher label；
- Episode evaluator；
- contact audit；
- state reset。

Reward：

保留四类：

1. insertion / alignment progress；
2. full-task success；
3. bad contact / penetration；
4. residual magnitude / smoothness penalty。

禁止仅用靠近距离判定最终成功。

沿用原生完整任务成功检测器和 Episode validator。

训练三个独立 seed，每个 seed 固定 32,768 steps。

训练结束后冻结模型。

不能根据 held-out 成功率挑选训练 checkpoint。

## 8. Stage F：正式对照实验

比较：

B0：Full Frozen Nominal。

B1：Frozen Nominal + V2 Local Residual，只在原来的 ALIGN / INSERT 区间介入。

B2：Full-Episode Post-Grasp Residual，GRASP_ESTABLISHED 后介入。

B3：Deterministic Post-Grasp XY Recovery。

B3 如果没有足够部署观测，不得使用 privileged pose 伪装成部署控制器；可以单独报告 Oracle 上界，但不得和 deployable baseline 混为一组。

所有方法使用：

- 相同初始状态；
- 相同扰动；
- 相同完整任务成功标准；
- 相同 Nominal；
- 相同安全边界。

64 个 held-out conditions。

20 个 nominal regression conditions。

三个训练 seed 全部独立测试。

使用 hierarchical paired bootstrap。

## 9. MVP GO 条件

必须全部满足：

1. V2 局部 Residual 多 seed 复现通过。
2. 完整 Nominal 重放通过。
3. 完整任务扰动后 Nominal 出现可测失败。
4. P0 或 P1 的 Observation Audit 通过。
5. Full-Task Residual 相对 B0 提升至少 15 个百分点。
6. 配对 95% CI 下界大于 0。
7. 至少 2/3 个 Full-Task PPO seeds 获得正向收益，且不存在单个 seed 的严重负迁移。
8. 无扰动 Nominal Regression 不超过 5 个百分点。
9. 相比 V2 Local Residual，Full-Task Residual 至少表现出不劣性；若要声称提前恢复有优势，必须报告其额外收益。
10. 无动作越界。
11. 无额外碰撞或不合理穿透增加。
12. Privileged Student Inputs = 0。
13. LLM Calls = 0。
14. Hardware Executions = 0。
15. Twin Mutations = 0。
16. 完整 Episode validator 通过。

如果统计区间不支持结论，标记 INCONCLUSIVE。

## 10. NO-GO 处理规则

不同失败必须分开记录：

- NOMINAL_REPLAY_FAILURE；
- OBSERVATION_UNIDENTIFIABLE；
- PERTURBATION_NOT_REPRODUCED；
- ACTION_INTERFACE_FAILURE；
- RESIDUAL_TRAINING_FAILURE；
- HELDOUT_GENERALIZATION_FAILURE；
- SAFETY_BOUND_FAILURE。

不得把所有失败都交给 Slow Agent。

不得自动新增模块。

不得改变本项目研究问题。

## 11. GitHub 交付

新增：

```text
docs/
  RESIDUAL_RESEARCH_CHARTER.md
  MINIMAL_RESEARCH_TASKBOOK_V3.md
  FINAL_EXPANSION_PLAN_V1.md

robot_agent/
  full_task_residual_env.py
  full_task_observation.py
  full_task_residual.py

scripts/
  audit_full_nominal.py
  audit_post_grasp_observability.py
  audit_in_hand_drift.py
  train_full_task_residual.py
  evaluate_full_task_residual.py
  run_full_task_residual.py

tests/
  test_full_reference.py
  test_full_task_observation.py
  test_post_grasp_perturbation.py
  test_full_task_residual_bounds.py
  test_student_privileged_isolation.py
  test_left_arm_invariance.py
  test_full_task_success.py

experiments/
  residual_fulltask_mvp/
    design.json
    reproduction.json
    observability.json
    perturbation_audit.json
    training_results.json
    heldout_results.json
    nominal_regression.json
    FINAL_MVP_REPORT.md
    source_manifest.json
```

保存模型到原 DATA_ROOT 的独立目录。

GitHub 只上传代码、配置、脱敏统计、模型哈希和报告。

禁止上传私有模型资产、原始视频、真实控制代码、密钥和原始大型数据。

每阶段 Commit 和 Push。

最终：

```bash
git status
git log -1
git push -u origin research/full-task-residual-20261124
git ls-remote origin refs/heads/research/full-task-residual-20261124
```

必须验证远端 HEAD 与本地一致。

## 12. 最终报告

报告必须分别回答：

1. V2 是否可复现？
2. 完整 Nominal 是否可重放？
3. 抓取后的位姿偏差是否能够被观测？
4. Proprio 是否足够？
5. 是否需要右腕 RGB？
6. Residual 能否在 LIFT / TRANSPORT 阶段恢复？
7. Residual 能否继续完成 INSERT？
8. Full-Task Residual 是否优于只在插入阶段修正？
9. PPO 是否优于确定性方法？
10. 当前仍有哪些 Sim-to-Real Gap 没有解决？

结论只允许：

GO / NO-GO / INCONCLUSIVE。

不得声称真实迁移成功。

本轮完成后停止，提交报告，不自动开始后续扩展阶段。

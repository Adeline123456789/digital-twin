# Event-Residual MVP V2：总体 NO-GO

本轮任务的实现、分阶段实验、消融与审计已完成。总体系统仍为 **NO-GO**，不能进入真实部署。分项结论必须分别读取：

| 范围 | 结论 | 直接证据 |
|---|---|---|
| 冻结 Nominal 局部重放 | PASS | 原生成功；完整积分状态误差 0 |
| 目标身份锁定/保持 | 初步可用 | 两个物理源 Episode 首帧锁定，6 帧遮挡均保留身份 |
| TOP_PIN / TOP_HOLE 联合视觉 | NO-GO | 独立验证中无可靠共同可见样本，语义校准未通过 |
| 本体感知 History Residual，局部几何扰动 | GO，仅限本轮仿真范围 | 16/32 对 4/32；+37.5 pp，95% CI [21.875, 53.125]；nominal 20/20 |
| 延迟/夹爪变化 curriculum 扩展 | INCONCLUSIVE | 7/32 对 6/32；+3.125 pp，95% CI [-15.625, 21.875] |
| 真实迁移、真实成功、双臂协同 | 未进行、未证明 | 零真实训练、零真实执行；双臂模型中右臂执行任务 |

40 mm 候选的快速变窄发生在指令基本恒定时，实际宽度变化约 2.06 mm，手内位移约 1.61 mm、旋转约 4.10°，接触几何集合变化；分类为 POSSIBLE_SLIP_OR_RESEATING。两指是否接触的布尔值保持不变，所以不能只靠“仍有接触”排除手内重排。原 nominal 最快变窄属于 CONTROL_COMMAND_CHANGE。此诊断使用仿真特权信息，不声称已有真实检测器，也未调摩擦。

动作使用 ±2 mm、3 mm/s、30 mm/s² 的每轴残差边界及临界阻尼平滑；实际评估中的修正和左臂隔离审计均通过。原生节奏保持 5.3 s。检查了仿真命令平滑性，未验证实机抖动。

硬件执行、真实训练、运行时 LLM、旧 Skill 执行、自动 Twin 更新均为 0。MVP-A 的失败没有阻断 MVP-B。V1 NO-GO 报告和原分支保留。未做 PPO 以外的自动升级、摩擦搜索、Twin 重建或成功阈值修改。

验证：180 tests passed，27 subtests passed；1 个旧测试因会导入本轮禁止读取的私有控制目录而排除，不能声称所有旧测试均已运行。图表已检查，模型/源码/轨迹哈希、84 个初态配对条件与原生冻结重放均有独立证据。

详细证据：

- [Perception report](PERCEPTION_REPORT.md)
- [Residual report](RESIDUAL_REPORT.md)
- [Requirement audit](COMPLETION_AUDIT.md)
- [Primary paired results](heldout_results.json)
- [Actuator-stage results](phase2_results.json)
- [Source manifest](source_manifest.json)

**Simulation-only。没有真实训练、没有真实执行、没有真实成功声明。**

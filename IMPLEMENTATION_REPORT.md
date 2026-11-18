# V2 快慢系统实施报告 · 2026-11-18

**离线诊断与有界技能回放可以使用。真机为 REAL_NOT_TESTED，物理根因仍为 UNKNOWN，5/5 验收仍待完成。** 从交接提交 `f3a0fe6` 建立本地分支 `feat/fast-slow-diagnosis`。本地 main 缺少后续 75 个提交。仓库原无 docs 任务书，已按用户指定的下载原件归档。交接前已存在的未跟踪 HTML 保留在工作区。

## 工程与实验预算

| 阶段 | 工程轮 | 新完整仿真用量 / 上限 | 主要实测与决策 |
|---|---:|---:|---|
| B | 1 | 0 / 2 | 原 unittest 126 项，119 通过、3 失败、4 错误。修正过时测试替身、保留原断言语义，并修复测试写回历史耗时文件的问题。pytest 基线 131 项通过。 |
| P0 | 2 | 0 / 0 | 对齐轨迹、H5、特权观测和原生报告。两条开发记录、六条冻结成功对照。 |
| P1 | 2 | 0 / 0 | 复用原五类记忆和 VS002。按版本过滤，NE011 可否决同源失败修复。索引首次生成因环境中缺 h5py 失败，已延迟导入并重新验证。 |
| P2 | 3 | 0 / 6 | 流式快路径、慢诊断预算、状态机和 JSON 技能图。历史 VS002 真实离线入口给出 −0.17855 mm 请求并停在新 Episode 后置条件。 |
| P3 | 4 | 0 / 12 | MuJoCo 和脱离夹爪的负对照。拒绝 −10° 倾角，接受窄范围的离线时钟校验。 |
| P4 | 2 | 2 / 4 | 原生接触读出、两个新完整仿真、同文件配对回放和独立复核。 |
| 合计 | 14 | 2 / 24 | 已用尽工程轮次，剩余 22 次仿真额度。真机执行 0 次。 |

每轮的假设、修改文件及哈希、测试数量、仿真与模型用量、用时、结果和 Git 回滚点保存在各轮的 design.json 与 RESULT.json。启动标记占用仿真预算，即使该次录制中断也会计数。历史实验的试验量保留原归属。[轮次账本](experiments/round_budget.json) · [预算复核程序](experiments/fast_slow_v2/audit_budget.py)

最终全量 pytest **153 项通过**。已核对外部模拟器源哈希清单、活动策略字节和两条新增 Episode 的完整文件布局。[最终核查](experiments/fast_slow_v2/P4.2/checks.json)

## 同源对照

原始夹持丢失在 **17.200 s**，第 515 帧首次跨过阈值，第 517 帧 **17.267 s** 确认。三个连续采样带来约 66.7 ms 确认延迟。18.300 s 为最后一个有时间戳的运输样本，之后原生报告记载 align 阶段 IK 异常，异常的精确时刻未知。另一条提前结束的同源开发轨迹返回 UNKNOWN。六个原版成功对照有 0 个误报、0 个 UNKNOWN，共四条唯一轨迹。唯一的已知丢失正例参与了开发，独立召回率为 NOT_MEASURED。[逐条配对结果](experiments/fast_slow_v2/P4.2/measurements.json)

新采集采用当前 original 源、双臂、socket3、相同初态和原生动作，两次完整录制都通过既有校验。各三路视频，每路 1,133 帧，回放物体位置最大误差均为 0 m。两份原始轨迹哈希相同，且与历史同配置轨迹相同。每组一次采集仅支持兼容性检查。对这两个新记录和八个历史记录分别运行了两种诊断，10/10 原生标签一致，10 份记录有 6 条唯一轨迹。全样本来自确定性和非随机分组，总体成功率区间、假成功率及真实稳定性为 NOT_MEASURED。未启动独立稳定性批次 V。[新采集设计](experiments/fast_slow_v2/P4.2/pair_design.json) · [原有故障开发集](experiments/research_stage4/round3/design.json)

采集组件耗时分别为 151.77 s 与 151.25 s。流处理共 9,995 帧，逐 Episode 的单帧处理 p50 为 0.0057–0.0096 ms，p95 为 0.0064–0.0120 ms，p99 为 0.0089–0.0309 ms。计时从已读取且对齐的结构化帧进入 FastLoop 开始，不包含曝光、磁盘读取和解码。新记录的完整 V2 诊断耗时 0.206、0.209 s，legacy 为 0.183、0.165 s。当前证据支持离线处理基准，无法推出实时硬件延迟、运动成功率收益或总成本下降。

本批运行时慢模型调用 0 次，输入和输出 token 均为 0。开发主 Agent、子 Agent 的 token 不计入运行时指标，成本未知。历史同范围 LLM 两次规划共 45,464 token，历史编译规则两次零调用。B 轮实测三个本地 CLI 版本进程启动耗时中位数 37.4 ms，完整模型 turn 基础上下文开销仍未独立测量。P1 轮历史 prompt 为 8,003 UTF-8 字节，当时的证据摘要为 2,201 字节；本批没有相同模型的新推理 token 对照。可选的本地慢模型经过协议替身和断开回退测试，现场端口未监听，本批未验证在线模型的诊断质量。[基线](experiments/fast_slow_v2/B1/measurements.json) · [P1 测量](experiments/fast_slow_v2/P1.2/measurements.json)

## 证据门槛与复核

旧的 −10° 候选虽有静态 IK 解，完整仿真观测到运输期间失去夹持。重新审查的决策为 REJECT_AND_ROLLBACK。时钟检查在一条开发轨迹和两条预先绑定的不同保留轨迹上保留原有成功，并识别内存中单独引入的 200 ms 时钟错位。ACCEPT 仅适用于原版 socket3 历史仿真的这一项时钟校验。两个结论的 global active 文件 SHA-256 相同，旧 VS002 仅可按其原 native 源哈希回放，当前源已变化。[最终决策](experiments/fast_slow_v2/P4.2/decisions.jsonl)

独立审查发现输入身份、回滚、采样率、统计重复和预算缺陷。修复已加入集成回归。第 515 帧的历史 XML 在当前 MuJoCo 中因惯量三角不等式无法重载，该帧动态接触重建为 DEFER_UNKNOWN；故障顺序来自录制的特权观测。已用独立的小型 MuJoCo 场景验证 inactive/active contact、间隙、weld 和 mocap 反例。历史真机缓存保留 RGB、编码器和原插入停止标签，实际接触力和三维插入姿态均未知。[独立复核](experiments/fast_slow_v2/P4.2/reviewer.md) · [重建限制](experiments/fast_slow_v2/P4.1/measurements.json) · [停止采集回读](experiments/fast_slow_v2/P4.2/real_replay.json)

## 使用与回滚

默认 `agent_loop.py` 继续使用 legacy。V2 使用 `--offline --fastslow --seed-episode ID`。逐轮回滚点保存在该轮 RESULT.json 的 rollback_ref，整个批次的基准为 `f3a0fe6`。本批仅有本地提交，原始模型、媒体、轨迹留在数据盘。完整快慢路径、角色、输入合同和命令见 [架构](docs/FAST_SLOW_ARCHITECTURE.md)、[证据合同](docs/EVIDENCE_SCHEMA.md) 和 [README](README.md)。

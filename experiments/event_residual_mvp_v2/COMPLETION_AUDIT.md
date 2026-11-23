# V2 逐项完成审计

审计目标是任务书 V2 的实现和实验执行是否完整；研究假设未通过时明确保留 NO-GO/INCONCLUSIVE。V1 的停止结论不改写为成功。

| V2 要求 | 当前证据 | 完成/结论 |
|---|---|---|
| 0：V1 基线/报告保留，新分支 | source_manifest.json 的 V1 报告哈希；远端 V1 仍为 46160b6 | 完成；本轮仅新分支 |
| 1：记录并修复 confidence deadlock | initial_audit；observation_adapter.py / interaction_observer.py；test_observation_tracking.py 的两种校准分支 | 完成；高 feature score 不等于语义可信 |
| 1：独立仿真验证阈值，Student 不读标签 | perception_design/dataset、observation_calibration.json、keypoint_detector.py 的 RGB-only predict | 完成；无阈值达标，保持 UNKNOWN |
| 2：身份与局部 cue 分离、四态跟踪 | target_lock.py、VisualTracker、实际两个 Episode 的遮挡回放、tracking tests | 完成；身份保持通过，局部联合 cue NO-GO |
| 3：REACQUIRE_OBSERVATION / contact mode / HOLD | recovery_router.py；test_observation_recovery.py、test_unresolved_no_action.py | 完成；只允许请求 HOLD，未新增搜索或硬件运动 |
| 4：left-wrist 不作部署必需 | perception_dataset、residual_env 的 external RGB 分支、source_manifest | 完成；left-wrist 不进入 Student |
| 5：训练前夹爪动力学审计 | audit_grasp_dynamics.py、grasp_dynamics_audit.json、4 张同步图；首个阶段 commit 89f709d 在 PPO 之前 | 完成；40 mm 候选为 POSSIBLE_SLIP_OR_RESEATING，非真实检测器 |
| MVP-A A1：global lock | 原始分辨率 RGB 两个 Episode 首帧锁定、已知布局方向限制 | 完成；2/2 是有限证据，不宣称广泛精度 |
| MVP-A A2：小型合成 heatmap 模型 | KeypointNet、keypoint_training.json；30 epochs、RGB only、投影/深度仅标签 | 完成；没有可支持联合 cue 的验证证据 |
| MVP-A A2：六类随机化 | build_perception_v2.py、perception_design.json；亮度/对比度/颜色纹理/视角/遮挡/模糊 | 完成；静态显露朝向增强单独标注为非物理轨迹 |
| MVP-A A3/A4：时间跟踪、可见率、误差、恢复、UNRESOLVED | observation_ablation.json、PERCEPTION_REPORT.md；按序列汇总并披露仅 2 个物理源 Episode | 完成；局部恢复未证明，UNRESOLVED 100% |
| MVP-B B1：Frozen base 真正执行 | V1 full replay + runtime_witness + phase2_default_invariance；同一 reference hash | 完成；局部 native success，积分状态误差 0 |
| MVP-B B2：先几何/手内 pose，再 delay/gripper | design.json、phase2_design.json、4 个 training manifest、phase2_interface_audit | 完成；未同时乱调质量/摩擦 |
| MVP-B B3/B4：0.5 s history、dropout、Student 隔离 | proprio_history.py；student_boundary_audit、runtime_access_audit、rgb_dropout_results | 完成；B2 为显式单帧基线，B3 无视觉仍执行；不宣称有用 RGB 融合 |
| MVP-B B5：紧凑策略、XY、真实注入和平滑 | residual_controller/env、test_residual_bounds、408 条正式动作审计、smooth_residual_example.svg | 完成；左臂/夹爪命令和动作边界通过 |
| MVP-B B6：只训练 residual PPO，稀疏/结构化分开 | residual_training.json；3×16,384 + 8,192 = 57,344 步；独立 checkpoint hashes | 完成；nominal 未训练 |
| MVP-B B7/B8：四基线、held-out 配对、bootstrap、nominal 回归 | heldout_results.json、phase2_results.json、initial_state_pairing.json（84 条件×3 模式） | 完成；几何 GO，执行器扩展 INCONCLUSIVE |
| 7：direct/inferred/privileged 分类 | interaction_events.EVENT_MODALITIES；审计中的 IN_HAND_POSE_DRIFT 标 privileged evaluator-only | 完成；未把潜在 inferred event 宣称为已训练真实检测器 |
| 8：五种比较语义和 UNKNOWN | 现有 comparator/三值逻辑/序列/deadline/UNRESOLVED 测试均保留通过 | 完成 |
| 9：Fast bounded path，Stopped-only Slow | frozen player/history/filter/native substep safety；131 个停止后 capsule；isolated MemoryRouter/Supervisor | 完成；运行时/离线自动 Slow LLM 均 0，无旧 Skill 执行 |
| 10：全部命名 V2 报告 | PERCEPTION_REPORT、RESIDUAL_REPORT、grasp_dynamics_audit、observation_ablation、residual_training、heldout_results、rgb_dropout_results、source_manifest、FINAL_V2_REPORT | 完成；分项结果分别报告 |
| 11：旧/新测试、发布、数据隔离 | 180 passed +27 subtests；明确排除 1 个会导入私有控制目录的旧测试；Git 新增文件检查 | 完成；最终 push/远端 HEAD 由提交后命令核验 |

## 结论边界

- 本轮没有开展 Real training、Real execution、Real success 验证或双臂协同。
- 几何 GO 不等于完整 Event-Gated 系统 GO；视觉校准失败，未新增依赖该视觉事件的 learned gate。
- 执行器 curriculum 的区间跨零，明确 INCONCLUSIVE；停止本轮训练和真实迁移考虑。
- 单训练 seed、小量物理视觉源、15 Hz 记录/30 Hz 控制等限制保留在报告中。
- `pelican-bicycle.html` 是原有未跟踪文件，本任务未修改或提交。

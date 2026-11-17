# 场景请求与实际模型一致性，2026-11-17

对应总任务第 7 节。发现：新 bounded 工具入口执行请求的新 holder shift，但给学生的事实只有旧 Episode；复用右臂任务 YAML 后切回 dual，原 `active_arms: [right]` 没被覆盖，可能产生声明双臂、实际单臂的不一致。

`learn.execution_task` 现在由采集与规划共用，显式解析任务路径、目标、实际机械臂和布局变化。`agent_challenge.execution_scene` 将当前请求与历史 Episode 分开，给出差异、坐标单位和最终配置哈希；相机/物理参数仅列变化字段，不向模型提供设备矩阵。`agent_loop` 与教学循环都传入本次请求的场景，每次反馈后重新比较实际执行场景。教师也看到该场景上下文。没有把原始 YAML、资产或模型上传。

核查了 [MimicGen 的物体参照转换源码](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/utils/pose_utils.py)（文件 Git blob `c31c4d107a9329d8aabb76ea2d91cf39d1cb6f7b`，`transform_source_data_segment_using_object_pose`）与 [生成/执行接口](https://mimicgen.github.io/docs/modules/datagen.html)。其方法保持子任务末端相对物体的关系，再映射到新物体位姿；本地 `try_plug_pick_sim.py:place_over_socket` 已使用当前夹爪—插头关系与插座目标，然后求解 IK。因此继续复用原实现，本次修复场景输入与信息传递，没有另写轨迹变换器、平移关节数组或引入额外平滑。

## 实际离线验证

使用最新完整仿真 `20261117T053833Z_25cf43e3` 的任务，构建现有 MuJoCo 模型，未运行物理步或控制硬件：

| 条件 | 实际状态维数/执行器 | 插头初始世界位移 | 场景差异 |
|---|---|---|---|
| 原右臂场景 | 7 / 7 | 0 | 无 |
| holder X 增加 1mm | 7 / 7 | 实测仿真几何 +1mm | 仅 holder_xy_m |
| 返回原场景 | 7 / 7 | 0 | 无，配置哈希恢复 |
| 同任务请求双臂 | 14 / 14 | 0 | 仅 active_arms，实际恢复 12 个左臂 body |
| 返回右臂 | 7 / 7 | 0 | 无，配置哈希恢复，左臂 body 为 0 |

5 次构建共 4.01 秒，右臂关节限位完全一致，原配置文件未修改。[原始数值](verification.json)记录源码和任务哈希；[相关回归](tests.json)覆盖实际入口收到场景、任务不累计偏移、双臂显式覆盖和隐私字段隔离。

**STATIC_SCENE_CONTRACT_ONLY / 使用仿真特权信息。** 这是配置与静态初始几何验证，未验证新轨迹、夹持/插入、完整旧→新→旧任务回归或真机效果。没有新增合格 Episode、模型调用或机械臂动作。静态 4 秒不等于端到端适配只需 4 秒。

现有 `llm` 默认规划器未晋升。共同配置解析已接入采集代码；新场景上下文进入可选工具规划和教学入口，但尚无新模型使用证据。总体目标仍按 [完整验收表](../continuation/REPORT.md)逐项推进。下一步需要在冻结场景与预算下执行学生/教师/独立布局测试，成功仿真逐条完成同参考真机对照；不得用本次静态构建替代这些验收。

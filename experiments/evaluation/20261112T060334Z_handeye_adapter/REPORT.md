# 接通既有 FK 与成熟 RGB 手眼求解器

2026-11-12；基线 `b6595576057613e151975ca6080353499861f5fe`。本次交付是离线数值接口及软件验证，不是第七轮真实数据校准。Stage 2 仍为六轮、未通过；Stage 3/4 未启动。没有真实运动、物理步进、XML 改写或模型重建。

## 实际复用与接入

继续针对 RGB 相机外参这一具体问题，阅读 [Bi-JCR §III-C，式 5–14](https://arxiv.org/html/2505.24819v1)，并检查 [OpenCV 4.5.4 `calibrateHandEyePark`](https://github.com/opencv/opencv/blob/4.5.4/modules/calib3d/src/calibration_handeye.cpp) 的相对变换、旋转求解与平移最小二乘，以及输入/输出坐标说明。采用其成熟单臂 AX=XB 求解器；没有重写优化算法或搬入整个双臂视觉框架。

环境核查首先发现：MuJoCo 使用的 OpenCV 5.0.0 未暴露 `calibrateHandEye`，另一个 MuJoCo Python 没有 cv2；但 `/usr/bin/python3` 已有 OpenCV 4.5.4、NumPy 和 SciPy，接口可用。直接使用现成系统环境，没有安装依赖或改动仿真环境。参考源码标签与实际库版本均为 4.5.4，不声称自行验证过发行包和上游二进制完全等价。

新增两个小接口：

- `prepare_rgb_handeye.py`：读取原有 camera.json 的 RGB 时间和 CSV 实测关节，调用原项目 `piper_sim.kinematics.forward` 及其现有 URDF，生成 link6 在 base_link 中的位姿。输出矩阵记号为 `T_base_gripper`，即将 link6 中的点变换到 base_link。
- `solve_rgb_handeye.py`：对配对机器人位姿和 RGB 目标位姿调用 OpenCV，只使用 train 拟合，保留原候选对比、heldout 一致性误差、运动约束奇异值、参数边界和失败报告。输出始终不晋升为活动标定。

已将真实多轴素材第 27、179、438、652、759 帧的机器人侧位姿准备到本地 `.local/rgb_handeye_adapter/multiaxis_robot_poses.json`。对应 entry/J6/J5/J4/return 保持阶段。这是已执行的档案读取与 FK 输入准备，不是新动作采集。**真实 RGB 目标位姿尚缺，输入明确标为 ROBOT_POSES_ONLY_RGB_TARGET_POSES_MISSING；没有用撤销的深度拟合结果填补。** 这些帧也尚未作为训练/独立验证划分；同次采集的返程不等于独立真实验证。

## 软件验证结果

使用解析相机投影生成六个平面目标点的 RGB 坐标，调用实际 PnP，再经进程接口调用实际手眼求解器。10 个合成姿态中 6 个训练、4 个留出。尺寸和外参在此验证中有已知真值，不意味着真实夹具也有相同精度的测量。

| 检查 | 实际结果 |
|---|---|
| 已知外参恢复 | 矩阵最大元素误差 1.55e-7，低于预定 1e-6 容差 |
| 合成留出位置一致性 | 原偏置外参 6.596 mm；候选约 4.07e-5 mm |
| 合成留出旋转一致性 | 原偏置外参 0.6743°；候选约 8.49e-6° |
| 仅改变留出目标观测 20 mm | 拟合矩阵逐元素不变；留出位置误差约 20.000 mm |
| 单轴旋转 | 拒绝不可辨识输入，未输出外参 |
| 输入声明毫米 | 拒绝，不猜测单位或自动换算 |
| 平移改变量超预设 1 mm 边界 | 拒绝，未放宽边界 |
| 缺少输入文件 | 保存 infrastructure_error JSON/Markdown，返回非零状态 |
| FK/时序/单位交叉检查 | 合成 CSV 的中间曝光对应半程实测关节，原 URDF FK 与现有 MuJoCo FK 在 1e-8 容差内一致；故意不同的目标关节值不影响结果 |

全套 **33 项测试通过**。详细数值见 result.json、case_results.json；verification.json 保存执行命令和日志哈希，run_checks.py 与 synthetic_input.json 可复现数值验证。合成投影没有噪声、遮挡和实际尺寸误差，极小残差只是数值正确性，**不得当成真实毫米级标定精度或采集成功率提升**。

## 保留与限制

软件接口保留；参数候选未用于实际相机或 MuJoCo。原项目 scripts/src/configs 的源码哈希与第六轮一致；直接加载既有 URDF/XML 作 FK 比较，未调用会重建模型的 resolve_model 路径。未把原项目源码、XML、原始 RGB、相机私人配置或真实派生位姿提交 Git。

接口文档见 [RGB_HAND_EYE.md](../../../digital_twin/calibration/RGB_HAND_EYE.md)。它还需要真实 RGB 对应/相机位姿、度量尺度依据、平面 PnP 歧义处理，以及独立真实验证。完整数值秩和小闭环残差都不能替代这些证据。没有宣称摩擦、质量、接触力或真实传输延迟已被辨识。

本次 planner 模型调用 0；研究编排 Token 未按轮计量。没有新的成功采集 Episode，因此有效成功率、每条数据 Token/总成本、人工干预改善仍未测。下一项有价值工作是完成这些既有 RGB 图像的可靠视觉位姿输入，不是继续增加无关综述或替换原仿真系统。

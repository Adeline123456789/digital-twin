# 既有 RGB/关节数据的手眼标定接口

状态：**软件验证通过，真实标定未运行**。没有新的物理参数或部署变更。适用于本机右腕相机、固定目标、刚性相机安装；不适用于直接混用其他机械臂示范的外参。六轴力和实测深度均不需要作为该求解器输入。

## 已接通的两段接口

在仓库根目录执行，下列 prepare 命令只读既有文件，不运行控制器或物理步进：

```sh
/usr/bin/python3 scripts/prepare_rgb_handeye.py \
  --capture /mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug/calibration/wrist_multiaxis_20261109_01 \
  --frames 27 179 438 652 759 \
  --output .local/UNIQUE_ID_robot_poses.json
```

它使用 RGB 曝光时间，插值 `commanded_probe.csv` 中的 `feedback_j1..6`，将度转换为弧度，调用原项目 `piper_sim.kinematics.forward()` 和现有 `assets/mujoco/kinematics.urdf`。输出 `T_base_gripper`，这里 base=URDF `base_link`，gripper=`link6`。不使用命令关节值，不混入双臂间距，不将 FK 当成外部末端实测真值。原相机配置和派生位姿保留本地。

prepare 输出明确标为 `ROBOT_POSES_ONLY_RGB_TARGET_POSES_MISSING`，**它不是可直接标定的完整输入**。还需要可靠的 RGB 目标位姿及其尺度依据，不能拿此前深度拟合结果填空。

完整输入字段由已提交的 [合成示例](../../experiments/evaluation/20261112T060334Z_handeye_adapter/synthetic_input.json) 演示：

- `T_base_gripper[i]`：末端坐标映射到基座坐标。
- `T_camera_target[i]`：目标坐标映射到 RGB 光学坐标；由已验证几何、内参与 RGB 对应估计，与机器人位姿按曝光配对。
- `translation_unit` 必须为 `m`；`split` 明确 train/heldout，留出数据不参与拟合。
- `baseline_T_gripper_camera` 是待比较的原外参；`bounds` 明确允许的平移/旋转改变量。原外参若只是标称 CAD 值，必须如实注明。
- `provenance` 保留来源 Episode/帧、尺寸依据、对应点与内参来源、时序与 FK 限制。

```sh
/usr/bin/python3 scripts/solve_rgb_handeye.py \
  --input .local/REVIEWED_POSE_INPUT.json \
  --output .local/UNIQUE_ID_handeye_candidate
```

调用系统 OpenCV 4.5.4 的 `calibrateHandEye(..., CALIB_HAND_EYE_PARK)`，输出 `T_gripper_camera`，含义是相机坐标到末端坐标。当前 MuJoCo 环境的 OpenCV 5.0.0 未暴露此函数；因此明确使用已有系统 Python，没有安装依赖或替换仿真环境。

## 证据边界

求解前拒绝静止/平行转轴造成的平移不可辨识；完整数值秩并不保证实际噪声条件下可辨识。求解后保留奇异值、参数改变量与边界判断。将固定目标经 `T_base_gripper @ T_gripper_camera @ T_camera_target` 映射回基座，报告 train/heldout 的位姿一致性；参考目标只从 train 构造。

一致性残差不是独立物理精度。平面 PnP 的双解、目标尺度、相机内参、关节零位、时间误差仍需在真实输入中处理。输入标注为米也不能证明数值确实是米；错误但一致的先验仍可能产生小残差。任何结果都保持 `calibration_promoted=false`，不会改 XML、活动配置或安全控制。

测试以解析投影生成 RGB 点，再调用 PnP 和实际手眼 API，核对已知变换、留出集污染隔离、退化、错误单位、超边界与失败记录。另以现有 MuJoCo 模型的 FK 交叉检查原 URDF FK、度/弧度及基座坐标。这些是软件验证，不能记成新真机成功或独立校准。

方法依据：[Bi-JCR §III-C 的 AX=XB 约束](https://arxiv.org/html/2505.24819v1)；实际复用 [OpenCV 4.5.4 Park–Martin 实现](https://github.com/opencv/opencv/blob/4.5.4/modules/calib3d/src/calibration_handeye.cpp)。未复制 Bi-JCR 框架或原仿真源码。

## RGB 输入提取的最新证据

[对应点检查](../../experiments/diagnosis/20261112T061643Z_rgb_correspondences/REPORT.md) 已在既有多轴 RGB 的四个帧对提取 28–30 个局部 LK 对应点。其前后向自一致性不等于真实几何精度；同张图像的已知变换检查揭示了局部点拟合向未观测区域外推的误差。新摘要报告凸包内支持范围，避免把整张单应映射当成观测。LoFTR/ECC 候选未晋升。

这些点仍不是 `T_camera_target`，不能直接填入本手眼接口。原项目任务记忆有夹具实物尺寸的历史记录，但测量误差来源未知；旧深度/CAD 拟合和遮挡角点不能补作独立真值。下一步的输入判断应基于局部对应、运动约束和不确定性，而不是因 SIFT 没有特征就断言 RGB 无信息。

运动诊断现已接入 prepare 与 solve：报告机器人侧秩、奇异值、条件数和线性误差放大上界，不新增未经验证的拒绝阈值。五帧多轴运动满秩，但返回帧接近初始姿态，不能充当充分独立验证。详见 [输入审计](../../experiments/diagnosis/20261112T065430Z_motion_observability/REPORT.md)。

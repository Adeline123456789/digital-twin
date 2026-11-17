# 第一轮：真实抓持失败

18mm/25mm/0.3Nm；完整仿真及冻结参考通过，真机53.71秒无模型暂停。抬升末夹爪47.5→34.6mm，RGB显示外壳相对夹爪改变，align触发原保护；最终插头掉在灰底座旁，用户独立确认夹不住。物理成功false，数据有效true，训练资格false。两路各1611帧、原始CAN/命令/图像和验证保留于complete_real/failed/20261117_real_grasp_six_rounds_000001。独立确认六电机和夹爪下使能、err_code0；用户已复原场景。

候选未晋升；回滚使用未修改原入口。实际私有控制版本见../controller_versions.json；sim/reference哈希和成本见本目录。该次完整失败计入六轮主验证第1轮，不因后续纠错清零。

抓取纠错第1轮改为分阶段验证：现有探针新增对象相对夹爪姿态，18mm原模型0.37°，point-contact敏感性130.57°；14mm170.09°、22mm0.21°。角度均为仿真特权量。22mm仅候选，原接触模型不修改、不以此宣称已识别真实摩擦。准备带独立物理回放的22mm抓取-抬升-折回诊断，先验证真实抓稳。

MuJoCo接触维数及旋转摩擦语义参考官方文档：https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters 。原探针只看抬高不充分；源码诊断改为保留姿态，已有相对姿态测试通过。

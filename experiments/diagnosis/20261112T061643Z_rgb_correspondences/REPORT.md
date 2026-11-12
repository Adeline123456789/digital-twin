# RGB 对应点：直接复用方法后的实际结果

状态：**获得局部 RGB 观测；完整区域的几何精度检查未通过；未进行真实参数标定。** 本次继续处理上一提交的具体缺口：手眼求解器已有机器人侧位姿，缺少可靠的图像观测。没有真机运动、MuJoCo 步进、深度输入或阶段晋升。Stage 2 物理校准轮数仍为六轮。

## 数据与被纠正的判断

读取既有 `calibration/wrist_multiaxis_20261109_01/right_wrist/`，参考帧 27，对应帧 179、438、652、759。原始 RGB、匹配点、图像遮罩和模型权重留在本地 `.local/20261112T061643Z_rgb_correspondences/`，公开文件记录哈希和摘要，没有写入正式 Episode 目录。

原项目任务记忆实际记录了灰色夹具 170×60×33 mm、白色夹具 212×76×30 mm，并称其为实物尺寸。因此不能继续笼统说“没有任何实物尺寸信息”。但所检查记录没有量具、误差范围或独立测量来源，当前只能视为历史尺寸先验。旧 `register_plug_fixture_views.py` 使用深度平面、遮挡角点外推和 CAD 锚点，不能直接当成独立 RGB 测量。源文件哈希见 result.json，没有复制原仿真源码。

## 直接相关的方法与实际接口

| 方法 | 原问题和已读内容 | 本次复用 | 适用条件与限制 |
|---|---|---|---|
| [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) | 尺度空间特征及可靠匹配；§3、§7.1 | [OpenCV 实现](https://github.com/opencv/opencv/blob/4.5.4/modules/features2d/src/sift.dispatch.cpp) 的 `SIFT_create`、双向比值匹配 | 需要可检测的局部结构；固定 ROI 的默认检测结果为零 |
| [LoFTR](https://arxiv.org/pdf/2104.00680) | 无检测器的粗到细局部匹配，处理传统检测器在低纹理区缺点；§3–4 | 读取 [原实现](https://github.com/zju3dv/LoFTR/blob/master/src/loftr/loftr.py) 及 [Kornia 0.7.3 接口](https://github.com/kornia/kornia/blob/v0.7.3/kornia/feature/loftr/loftr.py)，复用 indoor 权重 | 推理只需 RGB；预训练所用深度不能成为本机器人的测量。置信度不是像素真值 |
| [ECC](https://www.scitepress.org/PublishedPapers/2008/10872/10872.pdf) | 光度变化下的参数化图像对齐；§2–4 的目标函数、迭代和初始化限制 | 读取 [4.5.4 源码](https://github.com/opencv/opencv/blob/4.5.4/modules/video/src/ecc.cpp) 和 [4.13.0 源码](https://github.com/opencv/opencv/blob/4.13.0/modules/video/src/ecc.cpp)，直接调用 `findTransformECC` | 初始化和图像结构影响收敛；输入遮罩在目标图坐标。高相关也可能有几何偏差 |
| [金字塔 LK](https://robots.stanford.edu/cs223b04/algo_tracking.pdf) | 小局部窗口通过金字塔跟踪较大位移；§2.1–2.3 | 读取 [OpenCV 源码](https://github.com/opencv/opencv/blob/4.5.4/modules/video/src/lkpyramid.cpp)，直接导入仓库已有 `validate_rgb_motion.track_step` | 局部纹理、遮挡和光度变化影响跟踪。此前手指 grasp/lift 失败不能代表固定夹具多视图也失败 |

没有引入新的 Agent 框架。LoFTR 所需 Kornia 0.7.3 / kornia-rs 0.1.5 仅装在忽略目录 `.local/loftr_runtime`，使用已有 tote_insert 环境中的 PyTorch；没有修改仿真环境。权重来自 Kornia 指定的 [indoor checkpoint](https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt)，46,336,731 字节，SHA256 见结果文件；以 `weights_only=True` 读取。

## 结果与方法取舍

1. **SIFT 实际历史配对：0/4。** 固定遮罩包含 12,464 像素，但参考区域检测点为零。单独检查移除孔口排除区可恢复三个特征，仍不足以据此宣称整个物体可测量。没有在真实配对上不断调阈值。
2. **LoFTR 开发检查失败。** 对单张真实 RGB 施加已知图像变换，得到 166 对匹配；对应点误差中位 1.797 px、P90 3.790 px，未达预设 1/2 px 要求。随后既定 RANSAC 的全遮罩映射误差为 1.922/3.464 px；ECC 为 0.798/3.315 px，相关分数却达 0.9984。所有失败保留。
3. **固定四个后续软件变换比较：** 评价标准为全遮罩映射误差中位 <1 px、P90 <2 px。它们是已知图像变换，绝不是四次机器人任务。

| 图像变换 | LoFTR + RANSAC：P90 / 结果 | 再加 ECC：P90 / 结果 | 复用既有 LK：P90 / 结果 |
|---|---:|---:|---:|
| 平移 | 1.591 / 失败（中位超限） | 2.594 / 失败 | 0.002 / 通过 |
| 旋转缩放 | 1.812 / 失败（中位超限） | 3.331 / 失败 | 0.631 / 通过 |
| 透视 | 1.244 / 通过 | 1.411 / 通过 | 2.583 / 失败 |
| 透视及光度变化 | 2.697 / 失败 | 20.699 / 失败 | 0.300 / 通过 |
| 通过数 | **1/4** | **1/4** | **3/4** |

四变换在 ECC 检查前冻结；既有 LK 是之后加入的回顾性比较，不标为新的独立真实验证集。原始阈值和失败结果没有因后续分析而改写。**LoFTR/ECC 不晋升为默认标定输入。**

4. **历史真机图像的局部跟踪可继续利用。** 同一个既有 LK 在四对真实图像各保留 28、28、30、30 个点，前后向误差中位约 0.072、0.037、0.034、0.009 px。这是自一致性，真实对应精度仍 UNKNOWN，没有输出伪造的毫米级目标位姿。
5. **定位了此前没有区分的外推误差。** 三十个点集中在左侧，其凸包与遮罩交集只覆盖 32.96% 的遮罩。软件透视案例的实际跟踪点误差为中位 0.088 / P90 0.253 px；凸包内映射误差为 0.048 / 0.097 px，而外推到整个遮罩的 P90 是 2.583 px。这支持“局部观测不等于全区域测量”。这是事后误差定位，**不把原来的 3/4 改成 4/4，也不证明真实物理根因**。`observed_support()` 现已接入观测摘要，明确报告局部支持范围。

## 成本、代码与验证

SIFT 四对历史配对 0.153 s；LoFTR/ECC 四个软件案例含模型加载 14.466 s；既有 LK 四个软件案例加四对历史配对 0.022 s。这些是单次记录，OpenCV 版本、线程设置和加载范围不同，**不能当作公平的加速比，更不能当成成功 Episode 总成本降低**。未计入研究、安装及开发开销；planner 调用/Token 为 0，研究编排 Token 与人工干预没有可靠分项计量，留空。

新增小型提取、现成算法调用、支持范围计算及复现脚本。原 LK、任务程序、安全约束、速度和成功检测器均未改。全套 **38 项测试通过**，包含已知变换、光度变化、无纹理失败以及拒绝将远处像素标为已有局部观测的测试。增加支持范围字段后的报告重跑，逐项核对既有观测完全相同。原仿真 scripts/src/configs 哈希与第六轮一致。

JSON 文件保存各次成功、失败、版本与日志哈希。ER008 保存局部观测/外推区别，NE007 保存本配置未获益的证据；没有把其登记为真机操作 Skill 或接受的数字孪生校准。

## 复现与剩余缺口

从仓库根目录运行，输出使用新的本地目录名：

```sh
/usr/bin/python3 scripts/rgb_correspondences.py \
  --design experiments/diagnosis/20261112T061643Z_rgb_correspondences/design.json \
  --output .local/UNIQUE_sift

/media/midea/F496AC8096AC4548/piper_simulation/.venv-mujoco-hardware/bin/python \
  experiments/diagnosis/20261112T061643Z_rgb_correspondences/check_existing_tracker.py \
  --output .local/UNIQUE_existing_tracker

PYTHONPATH="$PWD/.local/loftr_runtime" /home/midea/miniconda3/envs/tote_insert/bin/python \
  experiments/diagnosis/20261112T061643Z_rgb_correspondences/check_registration.py \
  --output .local/UNIQUE_learned_registration
```

剩余问题已收窄：局部真实 RGB 对应点存在，但其尺度、三维目标对应、图像噪声传播及相机/机器人参数的可辨识性还没有完成验证。应利用这些实际观测与已有机器人运动约束继续判断，而不是把未知角点、未验证深度或高优化分数塞进手眼求解器。当前未运行新的真实参数拟合，未接受 MuJoCo 校准，未启动 Stage 3/4；整体研究目标仍未完成。

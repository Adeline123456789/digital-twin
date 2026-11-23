# V2 复核入口

在当前仓库运行，保持现有 simulator 和 DATA_ROOT。原始权重、RGB、轨迹和中间标签都在 `self_evolution/event_residual_mvp_v2/`，不属于新的完整正式 Episode。不要清空这些目录来重跑。

```bash
cd "/media/midea/F496AC8096AC4548/digital twin"
export PYTHONPATH=.local/event_residual_mvp_v2/runtime
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl
```

环境验证命令与独立 agent 复验记录见 `experiments/event_residual_mvp_v2/RUNTIME.md`。源码快照、版本和结果入口见 `source_manifest.json`。

```bash
/home/midea/miniconda3/envs/loop_act/bin/python scripts/verify_residual_runtime.py
/home/midea/miniconda3/envs/loop_act/bin/python scripts/verify_pairing_v2.py
/home/midea/miniconda3/envs/loop_act/bin/python scripts/summarize_residual_v2.py
/home/midea/miniconda3/envs/loop_act/bin/python scripts/plot_event_residual_v2.py
```

训练入口为 `train_residual_v2.py --variant B2_structured|B3_structured|B3_sparse|B3_actuator`；分别读取冻结的 design / phase2_design。前三个是独立固定预算训练，actuator 从指定 B3 structured 权重 warm-start。现有输出目录会拒绝覆盖；新的研究批次必须有新的设计和输出标识，不能把重新训练的模型替换进本批 held-out 结果。

评估入口为 `evaluate_residual_v2.py --variant ...`；`--phase2` 使用新的执行器条件，`--dropout 1` 验证缺失视觉，`--nominal-only` 做 nominal 回归。`_recheck` 记录用于源码预先快照后的重复核验，旧记录保留；统计中每个条件只计一次。

Perception 的顺序是 `build_perception_v2.py`、`build_keypoint_aux_v2.py`、`train_keypoint_v2.py`、`evaluate_perception_v2.py`。这些都是 simulation-only；首次错误孔位标签单独保留为 rejected，未进入训练或校准。完整 V1 报告保持原样。

本批验收结论是总体 NO-GO；仅局部几何 residual 为 GO，执行器扩展 INCONCLUSIVE。所有复核命令仍不构成硬件执行许可。

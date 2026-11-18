> 当前工作仅限离线会话交接。续作先读 [检查点](evolution/current_state.json)。真机尝试、物理参数搜索和原研究目标已暂停。历史目标与进展见[归档索引](evolution/checkpoints/20261118_session_handoff_history.md)。

> **2026-11-17历史归档：** 当时用户暂停研究。本批真机6/15条已执行，1成功、5失败（16.7%）；第4条误判已更正，第6条完整失败轨迹已归档。总体目标未完成，未接受物理校准或宣称成本下降。此状态覆盖下文历史进度，详见[当前采集报告](experiments/repeatability_cycle12/REPORT.md)和[逐条结果](experiments/repeatability_cycle12/episode_inventory.json)。本次只归档和发布，没有新增真机运动。

> 2026-11-16 第2周期：右臂已接回实际 `agent_loop.py --robot-arms right`；新增一条完整失败和一条完整成功Episode，1次模型调用完成已知任务修复，成功运动与原右臂程序逐帧一致。实际Token16,934，未证明模型/总成本收益；无真机运动。见[本周期报告](experiments/right_arm_adaptation/cycle2/REPORT.md)。当前策略LLM，旧compiled skill保留原适用范围。

> 2026-11-16：本批右臂场景适配及完整采集已通过限定仿真验收，见[右臂报告](experiments/right_arm_adaptation/cycle1/REPORT.md)。新入口`collect_current_scene.py --robot-arms right`，两相机、7维右臂状态。旧compiled skill因原生源码变更暂退出默认；未声称真实校准或模型能力提升。以下保留此前结果。

> 历史四阶段结果见 [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md)。当前 planner 为 `llm`；旧 compiled_alignment_v1 的源码适用范围保留，尚未迁移到右臂。`--arm A/B/C/D` 是实验组，`--robot-arms right/dual` 才选择仿真结构。完整采集按实际通道验收：双臂三相机，右臂外部与右腕两相机。低层 headless 输出不是完整数据集；没有自动接管外部生产 Agent 或真机。

# Failure-driven robot learning (additive prototype)

This repository contains a model-neutral, offline experiment layer for the existing Piper MuJoCo project. It does not replace MuJoCo, control hardware or train an LLM. Read PROJECT_AUDIT.md and protocol/robot_problem_solving.md first.

Existing roots can be selected using PIPER_SIM_ROOT and PIPER_DATA_ROOT. Raw videos, trajectories and images remain in the dataset; public reports contain relative episode references and SHA-256 hashes. The original source checkout is dirty: each run records source hashes, not merely its Git HEAD. That checkout is required to reproduce experiments; this repository is not a standalone simulator distribution.

```sh
# Use the existing environment that has mujoco, av, scipy and yaml.
"$PIPER_SIM_ROOT/.venv-mujoco-hardware/bin/python" scripts/learn.py --kind baseline --offset-mm 2
"$PIPER_SIM_ROOT/.venv-mujoco-hardware/bin/python" scripts/learn.py --kind improvement --offset-mm 0
python scripts/learn.py --retrieve plug_not_seated --repair hold_gripper_during_retreat
python -m unittest discover -s tests -v
python scripts/publish.py --message 'Record completed experiment' experiments/baseline/<ID>
```

The retrieval CLI supplies at most five matching records. Existing Agent orchestrators must call it and follow the shared protocol; no external Agent prompt has been silently changed. There is no unattended hardware execution or background self-modifying service. Evolution runs are explicit, bounded batches with independent regression gates and version records.

Publication uses explicit paths, size/type checks and heuristic secret scanning plus human/agent review. These checks cannot guarantee arbitrary new input contains no private information. Do not publish raw recordings, credentials or logs containing them. Push failures retain local commits and write details under ignored .local/.

Updated storage rule: see AGENTS.md for the user-prescribed exact data roots. New headless trials archive their source evidence by observed outcome; they do not claim to have recorded videos. Successful recordings use `complete_sim/succeed/<episode_id>/` with their own `videos/`, `proprio_stats/` and `outcome.json`, matching the failure layout. The original Episode and its three videos are preserved. Headless source-only success evidence stays under `self_evolution/archived_trials/`. Historical split paths can be resolved using the storage-migration manifest in `experiments/maintenance/success_episode_layout/`. The CLI now commits/pushes each completed simulation report by default, including failures; use `--local-only` for local validation. Model-evaluation batches are published after review as a round. Original simulation code and all raw media remain outside Git.

Actual model execution (no hardware):

```sh
"$PIPER_SIM_ROOT/.venv-mujoco-hardware/bin/python" scripts/agent_loop.py --robot-arms right --arm D --shift-mm 0
# Known-task deterministic collection uses no planner call:
"$PIPER_SIM_ROOT/.venv-mujoco-hardware/bin/python" scripts/collect_current_scene.py --robot-arms right
# Select --robot-arms dual to restore the original simulation.
```

The planner returns bounded JSON and the adapter physically executes the existing simulation script. A/B/C freeze the first plan; D may revise from actual feedback. Reports keep the model calls and physics trials separate. `scripts/agent_challenge.py` additionally tests a synthetic hidden target-frame error using simulator privileged precontact observations; it is not real visual calibration. Historical scene/source versions must not be pooled.

A newer preserved simulator tree was also found at `$PIPER_DATA_ROOT/calibration/offline_workspace_20261110`. `scripts/run_retained_scene.py` invokes its documented final-scene command without editing or uploading it; its results are reported separately.

Use the retained final scene with the Agent (existing offline environment; no simulator upload):

```sh
PIPER_SIM_ROOT="$PIPER_DATA_ROOT/calibration/offline_workspace_20261110" \
PIPER_SIM_PROFILE=retained_final \
"$PIPER_DATA_ROOT/calibration/offline_env/bin/python" scripts/agent_loop.py --arm D --shift-mm 0.25
```

This profile preserves the documented grasp roll/opening/alignment settings and 25°/s free-space speed; the existing script still caps contact phases at 12°/s. Set `ROBOT_LLM_MODEL` to another available Codex model to use the same planning schema, protocol and memory. No model receives shell/CAN/XML control through this adapter. Physics profiles are not auto-calibrated or silently promoted.

For this workstation, initialize the paths used above explicitly:

```sh
export PIPER_DATA_ROOT=/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug
export PIPER_SIM_ROOT=/media/midea/F496AC8096AC4548/piper_simulation
```

See ROUND2_REPORT.md for executed model-feedback results and COMPLETION_AUDIT.md for what was completed versus explicitly deferred. Raw-force and real-success claims are not inferred from simulator or motor telemetry.

Dataset-format correction: failed episodes are now recorded through the existing three-camera collector and checked against the user's reference schema before being placed in `complete_sim/failed/`. This can take several minutes. Headless `source_trial` directories alone are not published as dataset episodes. If recording or validation fails, the formal episode is not created; diagnostic evidence remains outside that directory. The user-requested deletion of the 14 earlier incomplete folders is documented under `experiments/format_migration/`.

Stage 3 collection result: an opt-in `learn.simulate(..., record_inline=True)` records through the native writer during the original physics run; use `finalize_recordings.py <ID> --stage self_evolution/stage3_recording_<ID>` to verify/install it. Default collection remains unchanged. See [round2](experiments/research_stage3/round2/REPORT.md) for paired motion/format verification, timing, failed pixel gate and transparent measurement correction. This uses simulator privileged information and does not establish real calibration or model capability gains.

## Offline Fast/Slow V2

Use the existing MuJoCo Python environment. The following commands replay local recordings and print JSON.

```sh
python scripts/agent_loop.py --offline --fastslow --local-only --seed-episode 20261112T090023Z_1341a28d
python scripts/triage_episode.py 20261112T075231Z_2a125acc --fastslow
python scripts/triage_episode.py 20261112T090023Z_1341a28d --legacy
python experiments/fast_slow_v2/audit_budget.py
```

`agent_loop.py` without a mode flag retains legacy collection behavior. `--offline` selects recorded data; adding `--fastslow` runs the V2 supervisor. The standalone triage entry defaults to V2 and always stays local. Legacy collection's existing `--local-only` controls publication. The optional `triage_episode.py --local-llm` permits one bounded diagnosis using an existing loopback llama.cpp server. `--contact-frame FRAME` reads a saved simulation frame. For recorded real data use `--stopped-real-capture PATH --cache DIR`; only stopped captures can be summarized.

[Architecture](docs/FAST_SLOW_ARCHITECTURE.md) · [Evidence schema](docs/EVIDENCE_SCHEMA.md) · [Batch results](IMPLEMENTATION_REPORT.md)

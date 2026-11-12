> 最新结果：四阶段有界研究完成，见 [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md)。完整系统仍NO-GO；当前只验证了限定仿真组件。Agent D默认优先执行适用的已验证Skill；不适用时用LLM。用 `--planner-policy llm` 回滚。所有新完整采集请使用下方 `scripts/agent_loop.py` 入口及已有MuJoCo环境；它核验三路视频、状态/动作并保存成功/失败。`learn.py` 的低层headless示例仅用于诊断，不应把其原始输出当作完整数据集。没有自动接管外部生产Agent或真机。

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
"$PIPER_SIM_ROOT/.venv-mujoco-hardware/bin/python" scripts/agent_loop.py --arm D --shift-mm 0.5
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

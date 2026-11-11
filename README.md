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

Updated storage rule: see AGENTS.md for the user-prescribed exact data roots. New headless trials archive their source evidence by observed outcome; they do not claim to have recorded videos. New successful videos use experiment-ID subdirectories beneath the existing `20261111_sim_calibrated_pick_insert_000001/videos/`, preserving its three original videos. The CLI now commits/pushes each completed simulation report by default, including failures; use `--local-only` for local validation. Model-evaluation batches are published after review as a round. Original simulation code and all raw media remain outside Git.

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

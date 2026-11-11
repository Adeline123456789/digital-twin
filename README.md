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

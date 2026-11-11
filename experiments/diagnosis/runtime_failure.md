# Runtime failure diagnosis

The first new simulation (20261111T083553Z_11b23ba4) failed before any physics steps: importing mixed_recording raised ModuleNotFoundError for av in .venv-mujoco. This is not a robot task failure and is excluded from physical success denominators. Existing .venv-mujoco-hardware imports av, mujoco, scipy and yaml successfully; reuse it without installing dependencies. No hardware module is imported by the trial, despite the environment name.

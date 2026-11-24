# Full-Episode Post-Grasp Residual Research Charter

## Question

Can a simulation-trained residual policy use deployable observations to recover a frozen full-task nominal after the plug moves relative to the gripper after `GRASP_ESTABLISHED`?

The first full-task scope is `LIFT` through `RETREAT`. It does not train a residual for `PRE-GRASP` or `GRASP`.

## Fixed boundaries

- Simulation only; hardware execution, real-data reads, real training and runtime LLM calls are zero.
- The nominal reference, MuJoCo friction/contact solver and native success criteria remain frozen.
- The active task arm is right; the left arm follows its frozen nominal commands.
- Student inputs are deployable observations only. Privileged simulator state is limited to reset, reward and offline evaluation.
- The robot has no six-axis force/torque sensor; none is represented in observations or claims.
- Raw trajectories, video and model weights stay under `DATA_ROOT`; Git contains code, hashes and reviewed statistics.

## Evidence rule

Stages execute in order and only a `GO` advances. A `NO-GO` or `INCONCLUSIVE` result stops later training. Counterfactual post-grasp state perturbations and contact-mediated drift are reported separately. No result from this branch establishes real-robot transfer.

The executable gates and deliverables are defined in [MINIMAL_RESEARCH_TASKBOOK_V3.md](MINIMAL_RESEARCH_TASKBOOK_V3.md).

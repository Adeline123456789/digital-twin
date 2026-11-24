# Core Experiment Execution Amendment V2

## Purpose

Auxiliary diagnostic performance no longer blocks the core Residual experiment unless it invalidates experiment integrity. Preserve all prior Stage A/B/C/D artifacts and conclusions; this amendment changes progression semantics, not history.

## Current interpretation

- **Stage B:** VALID. The complete Frozen Nominal exists and is executable.
- **Stage C:** VALID diagnostic. P0 detects perturbation presence with AUROC 1.0. Its four-direction classifier score of 0.50 is not a hard prerequisite because the Residual Policy is not required to receive an explicit direction class. P0 is authorized as the Student interface.
- **Stage D:** The direct-qpos family is valid but too weak as the primary recovery benchmark because all four boundary Frozen-Nominal rollouts succeed. Existing force probes caused measurable drift but did not provide a direction-controllable family. Direction controllability is not required for training.

## Hard stop conditions

Only these conditions stop the core experiment:

1. Frozen Nominal execution is invalid.
2. Student receives privileged simulator state.
3. Safety or action bounds are violated.
4. Train/test leakage or provenance corruption occurs.
5. A perturbation immediately destroys task semantics instead of producing a valid recoverable state.

Do not add observation classifiers, Agents, memory modules, sensors, or Twin calibration before the core experiment.

## CONTACT_MEDIATED_DRIFT_V1

Use unchanged MuJoCo physics and the frozen grasp. During LIFT or TRANSPORT, apply a short bounded external disturbance to the plug. Do not modify friction, mass, contact solver, success detector, or Frozen Nominal.

Pilot domain:

- continuous XY force angle;
- force magnitude initially 5–25 N;
- duration 50–200 ms;
- injection at lift-middle, transport-early, or transport-late.

Record actual plug-in-gripper translation and rotation, gripper aperture, contact topology, max penetration, and final outcome. Requested and actual drift directions need not agree; the actual physical result defines the perturbation.

Select a valid challenge distribution with both Frozen-Nominal successes and failures, preferably 20%–80% success. Reject only collision-bound violations, immediate task-semantic destruction, or provenance loss. Freeze `contact_drift_design.json` before policy training and held-out generation.

## Student and training

Use only P0: 15-frame right joint position, right joint velocity, Frozen-Nominal action, controller-feedback residual, gripper width, and phase history. Exact object/socket pose, contact pair, penetration, success, requested force direction, and actual privileged drift are forbidden Student inputs. Privileged state is limited to reward, reset, diagnostics, and final evaluation.

Train three independent fixed-budget B2 History PPO Structured Reward seeds with the previously validated XY Residual bounds:

- position ≤2 mm per axis;
- speed ≤3 mm/s;
- acceleration ≤30 mm/s²;
- joint correction ≤0.03 rad.

Do not tune against held-out results.

## Evaluation

Evaluate B0 Frozen Nominal, B1 existing deterministic recovery/search, and B2 History PPO on identical newly generated held-out disturbances. A privileged Oracle may be reported only as a separate upper bound.

Report full-task success, recovered B0 failures, spoiled B0 successes, collision/safety failures, residual magnitude, correction variation, seed variance, and paired confidence intervals. Do not impose a new arbitrary percentage-point gate. Classify the final finding as `STRONG_SUPPORT`, `LIMITED_SUPPORT`, `NOT_SUPPORTED`, or `INVALID`; only `INVALID` automatically stops research.

No Slow Agent work is authorized in this batch.

## Deliverables

- `contact_drift_pilot.json`
- `contact_drift_design.json`
- `contact_drift_training.json`
- `contact_drift_heldout.json`
- `CONTACT_DRIFT_RESIDUAL_REPORT.md`

Continue on `research/full-task-residual-20261124`. Commit and push before and after the core experiment.

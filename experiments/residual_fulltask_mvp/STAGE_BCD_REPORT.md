# Full-Task Residual Stages B/C/D

## Decision

`HOLD before Stage E — OBSERVATION_UNIDENTIFIABLE.`

Stage B and Stage D are valid. Stage C is a valid audit, but neither deployable candidate meets the frozen four-direction threshold, so Stage E has no qualified observation mode and was not started.

## Stage B — Frozen Nominal

- **VALIDITY:** VALID
- **HYPOTHESIS:** SUPPORTED
- **PROGRAM_DECISION:** CONTINUE
- Existing frozen source reference: 1,133 frames, 14-D dual-arm SI targets, historical planner-free replay PASS, full Episode validator PASS.
- `APPROACH` is a task-level label for native `descend`; the source has no separate native approach phase.
- The exported JSONL is an exact schema projection. Historical execution consumed the source frozen reference, not the exported JSONL itself.

## Stage C — Observation Audit

- **VALIDITY:** VALID
- **HYPOTHESIS:** NOT_SUPPORTED
- **PROGRAM_DECISION:** HOLD
- **FAILURE_CATEGORY:** OBSERVATION_UNIDENTIFIABLE

The corrected audit uses raw P0/P1 histories, separately executes each nominal/perturbed pair, splits by complete pair and disjoint amplitude fractions, and labels direction from both the controlled treatment and the paired current plug/gripper error. No paired-nominal subtraction is part of the classifier input.

This revision supersedes the initial Stage C artifact in commit `9cdb107`, whose shared-nominal subtraction and treatment-only direction label overstated deployable performance. The corrected four-direction score is the minimum of balanced treatment accuracy and macro recall for current-error direction; the frozen 0.60 threshold is unchanged.

| Mode | Detection AUROC | Four-direction score | Current macro recall | Treatment direction | Qualified |
|---|---:|---:|---:|---:|---|
| P0 proprio history | 1.00 | 0.500 | 0.500 | 0.500 | No |
| P1 P0 + right-wrist RGB | 1.00 | 0.477 | 0.477 | 0.500 | No |

The direction threshold is 0.60. P0 recognizes X directions but has zero recall for the two current Y classes. P1 also fails direction qualification and remains non-deployable because simulator-to-real right-wrist viewpoint correspondence is unverified.

Allowed claim: P0 detects the presence of these deterministic simulated interventions. It does not identify all four directions at the required accuracy, and this audit does not establish real-noise or real-transfer robustness.

## Stage D — Perturbation Audit

- **VALIDITY:** VALID
- **HYPOTHESIS:** SUPPORTED for constructing a controlled state-perturbation family
- **PROGRAM_DECISION:** CONTINUE at the Stage D gate
- **NOMINAL-FAILURE HYPOTHESIS:** NOT_SUPPORTED

Frozen cardinal maxima in the right-gripper frame are `+X 0.18 mm`, `−X 0.50 mm`, `+Y 0.18 mm`, and `−Y 0.18 mm`. All four boundary full-task replays stay within the 0.20 mm penetration bound and all four complete successfully under Frozen Nominal. Therefore the family is physically valid in the simulator, but it supplies zero task-failure recovery cases.

External-force probes without friction changes did not produce a valid direction-controllable four-way slip family: `PHYSICAL_SLIP_NOT_REPRODUCED`. The controlled qpos edit is not claimed to be physical slip or a measured real-world distribution.

## Execution boundary

- Stage E training steps: 0
- Hardware executions: 0
- Real-data reads/training: 0
- Runtime LLM control: 0
- Friction/contact-solver/success-threshold changes: 0
- Automatic Twin mutations: 0

Evidence: `full_task_reference_manifest.json`, `observability.json`, `perturbation_pilot.json`, `perturbation_design.json`, and `perturbation_audit.json` in this directory.

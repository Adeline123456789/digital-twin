# Research Gate Amendment V1

## Purpose

This amendment changes research progression semantics, not the scientific result of Stage A.

Do not rewrite, delete, or relabel the preregistered Stage A result. The historical result remains:

`Stage A NO-GO under the preregistered >=15 percentage-point effect-size target.`

The research-program continuation decision is separately:

`CONTINUE — positive Residual evidence supported; 15 pp stretch target not met.`

## 1. Why continuation is allowed

Observed Stage A:

- B0 Frozen Nominal: 9/64 = 14.0625%.
- B1 Deterministic Search: 20/64 = 31.25%.
- B2 History PPO, five-seed mean: 83/320 = 25.9375%.
- B2 − B0: +11.875 pp.
- Hierarchical 95% CI: [+0.625, +23.4375] pp.
- Four of five training seeds improve over B0.
- Nominal regression: 100/100 successes.
- Privileged Student Inputs: 0.
- Action bounds and left-arm invariance: PASS.
- Hardware, real-data, real-training, runtime LLM, and automatic Twin mutation: 0.

Therefore:

- `Residual > Frozen Nominal` is currently **SUPPORTED**.
- `Residual improvement >=15 pp` is **NOT_SUPPORTED**.
- `PPO > Deterministic Search` is **NOT_SUPPORTED**.

Stage A must not be rerun or tuned to chase the 15 pp target.

## 2. Gate taxonomy

### HARD VALIDITY GATE

Failure invalidates downstream interpretation and stops dependent stages. Examples include an unexecutable frozen nominal, privileged Student leakage, action or safety bound violation, train/test leakage, paired initial-state mismatch, success-detector mutation, corrupted provenance, or an invalid observation interface.

### CLAIM GATE

Controls which scientific claim may be made. Failure does not automatically terminate unrelated downstream experiments. Examples include PPO beating deterministic search, Event-Gated beating Phase-Gated, or a particular effect exceeding a chosen magnitude.

### TARGET / STRETCH GOAL

A desired effect magnitude. Failure means the target was not achieved; by itself it does not terminate the research program.

The previous 15 pp Stage A threshold is reclassified as a target/stretch goal for future interpretation. This does not retroactively change the preregistered Stage A result.

## 3. Stage result vocabulary

Every future stage reports separately:

- `VALIDITY`: `VALID` / `INVALID`;
- `HYPOTHESIS`: `SUPPORTED` / `NOT_SUPPORTED` / `INCONCLUSIVE`;
- `CLAIMS_ALLOWED`;
- `CLAIMS_NOT_ALLOWED`;
- `PROGRAM_DECISION`: `CONTINUE` / `HOLD` / `TERMINATE`.

Only `INVALID` normally stops all dependent work. `NOT_SUPPORTED` stops only the unsupported claim unless the next experiment logically requires that claim.

## 4. Correct dependency graph

Stage A and Stage B are not strict serial scientific dependencies.

- Stage A: Local Residual multi-seed evidence.
- Stage B: Complete Frozen Nominal audit.
- Stage C: Post-grasp observability.
- Stage D: Post-grasp perturbation construction.
- Stage E: Full-task Post-Grasp Residual.

Stage E requires all of:

`Stage B VALID` **and** `Stage C provides at least one deployable observation mode` **and** `Stage D provides a valid perturbation family`.

Stage A supplies baseline evidence and comparison methods, but a missed stretch target does not block B, C, or D.

## 5. Immediate next work

Do not retrain Stage A. Proceed with the already-defined V3 tasks:

### Stage B — Complete Frozen Nominal Audit

Verify the full APPROACH → GRASP → LIFT → TRANSPORT → ALIGN → INSERT → RELEASE → RETREAT executable reference. Make no new planner calls or privileged replanning after freeze.

### Stage C — Post-Grasp Observability Audit

Test whether controlled post-grasp plug/gripper displacement is distinguishable using:

- P0: q, dq, nominal action, controller-feedback residual, gripper width, and phase/history.
- P1: P0 plus right-wrist RGB plug-to-gripper relative features.

Do not require Top Pin plus Top Hole visibility.

### Stage D — Perturbation Validation

Build the already-predeclared controlled post-grasp XY displacement experiment. Keep controlled counterfactual displacement separate from physically contact-mediated slip. Do not modify friction.

## 6. Stop conditions for B/C/D

- Stage B is `INVALID` if a full frozen nominal reference cannot be faithfully executed without replanning. Full-Task Residual then waits until the nominal interface is repaired.
- Stage C is `NOT_SUPPORTED` if neither P0 nor P1 distinguishes the perturbation. Do not train a Student to infer unavailable information; record `OBSERVATION_UNIDENTIFIABLE`.
- Stage D is `INVALID` if the intended perturbation does not physically modify task state or breaks task semantics. Do not proceed to training.

## 7. Fixed research objective

Can a simulation-trained Residual Policy use deployable observations to recover a Frozen Nominal from post-grasp/full-task deviations without rebuilding the Twin for every failure?

Do not add new Agents, memory systems, sensor modalities, reward terms, or Twin calibration routines solely because the target effect size was missed. Agent evaluation remains a later, separately tested component.

## 8. Repository record

Keep branch `research/full-task-residual-20261124`. Preserve `FINAL_MVP_REPORT.md`, all five checkpoints, all Stage A evaluations, and the original preregistration hashes. Do not overwrite history.

Commit and push this amendment before starting Stage B. At the end of B/C/D, publish separate validity, hypothesis, claims, and program-decision fields. Do not automatically start Stage E unless its true hard prerequisites are valid.

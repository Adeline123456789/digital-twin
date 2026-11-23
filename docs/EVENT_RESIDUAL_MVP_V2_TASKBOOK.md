# Event-Residual MVP V2 — 修复 Observation Deadlock 并独立验证 Residual RL

## 0. 基础

Repository:

`https://github.com/Adeline123456789/digital-twin`

Current completed MVP branch:

`research/event-residual-dual-20261123`

Frozen baseline commit:

`46160b6200dff7add154d7b672f46d4bc0d1ea14`

Preserve the existing NO-GO result.

Create:

`research/event-residual-dual-v2-20261123`

Do not rewrite or delete the V1 MVP report.

This batch remains:

- Simulation-only
- zero hardware execution
- zero runtime LLM control
- zero automatic Twin mutation

Use the dual-arm MuJoCo model, but keep the existing task semantics: the right arm performs the plug task; do not claim bimanual manipulation.

---

# 1. First fix the V0 logical deadlock

Audit:

- `robot_agent/observation_adapter.py`
- `robot_agent/interaction_observer.py`

Current V0 assigns at most:

`visual.confidence = 0.5`

while `alignment_event()` requires:

`confidence >= 0.9`.

Therefore a visual candidate cannot become TRUE/FALSE even when both candidate features exist.

Record this explicitly as:

`OBSERVATION_CONFIDENCE_CONTRACT_DEADLOCK`.

Do not simply change 0.5 to 1.0.

Instead:

1. Remove hard-coded semantic confidence from the detector.
2. Separate:
   - `feature_score`
   - `semantic_validation_status`
   - `visibility`
   - `tracking_uncertainty`
3. Determine the event threshold using a separate simulation validation set with privileged labels available only to the evaluator.
4. The deployed observer must not read those privileged labels.

---

# 2. Separate Target Identity from Local Alignment

Delete the runtime requirement:

“every align/insert frame must show one USB + three AC panels”.

Replace with a state machine:

`UNLOCKED -> TARGET_LOCKED -> LOCAL_TRACK -> OCCLUDED_CONTACT`

## TARGET_LOCKED

Before close insertion, identify outlet 3 once.

This may use:

- external RGB;
- board layout;
- known task target ID;
- temporal visual tracking.

Target ID is persistent after lock.

Losing the full board from the image must not erase target identity immediately.

Store:

- target ID;
- last valid target keypoints;
- timestamp;
- tracking uncertainty.

## LOCAL_TRACK

Near the socket, stop requiring the whole AC module.

For the current task define only the deployable local cue:

`TOP_PIN ↔ TOP_HOLE`

Measure:

`e_img = [u_hole-u_pin, v_hole-v_pin]`

Use only features that can also plausibly exist in current real RGB.

Do not require the lower pins or lower holes.

## OCCLUDED_CONTACT

When the required RGB cue is unavailable:

- do not label the physical event FALSE;
- mark the visual observation UNKNOWN;
- allow the controller to switch to the contact/proprioceptive observation mode.

---

# 3. Add observation recovery

Extend `RecoveryRouter` with:

`REACQUIRE_OBSERVATION`

The available routes become:

- NO_ACTION
- RETARGET
- RESIDUAL
- REACQUIRE_OBSERVATION
- STRUCTURAL_CANDIDATE
- UNKNOWN

Policy:

### Pre-contact UNRESOLVED

If target/local visual evidence is missing while the robot is still in a safe pre-contact state:

`UNRESOLVED -> REACQUIRE_OBSERVATION`

First V2 implementation may:

- hold;
- retreat to the previously validated preinsert observation pose;
- re-observe.

Do not perform uncontrolled search motions.

### Near/after-contact UNRESOLVED

Do not repeatedly retreat only to make the image visible.

Switch to:

`PROPRIOCEPTIVE_CONTACT_MODE`

The safety supervisor retains final HOLD/STOP authority.

---

# 4. Do not use the left-wrist camera as a required deployment observation

The dual-arm simulator may retain all three cameras for evaluation.

However, the current physical deployment scope is right-arm hardware.

Therefore:

Student/deployment-required observations must use only modalities available on the intended real system.

Treat left-wrist RGB as:

`SIMULATION_AUXILIARY_EVALUATOR_ONLY`

unless separately justified as real-deployable later.

---

# 5. Diagnose the sudden gripper narrowing before training

Implement:

`scripts/audit_grasp_dynamics.py`

For the original nominal and the user-inspired 40 mm candidate, export synchronized traces:

1. commanded gripper aperture;
2. simulated actual gripper aperture;
3. plug position relative to right gripper;
4. plug orientation relative to right gripper;
5. left-finger/plug contact;
6. right-finger/plug contact;
7. phase;
8. controller target;
9. penetration.

Plot/record the one-second interval where the apparent aperture changes rapidly.

Classify only from evidence:

### CONTROL_COMMAND_CHANGE

if the commanded target itself changes.

### POSSIBLE_SLIP_OR_RESEATING

if command is approximately constant while:

- actual aperture changes;
- plug-in-gripper transform changes;
- contact topology changes.

### VISUAL_PERSPECTIVE_ONLY

only if recorded physical aperture and plug-in-gripper transform remain stable.

### SIM_CONTACT_ARTIFACT_CANDIDATE

if the change depends on implausible penetration/contact-solver behavior.

Do not tune friction in this batch.

Add a report:

`experiments/event_residual_mvp_v2/grasp_dynamics_audit.json`

If possible, add the event:

`IN_HAND_POSE_DRIFT`

for privileged simulation evaluation.

Do not yet claim a real detector for this event.

---

# 6. Split the research into two independent MVPs

The V1 mistake was making visual observation a hard prerequisite for Residual RL.

V2 must run two independent tracks.

---

# MVP-A — Deployable Perception

Goal:

demonstrate that RGB can provide useful pre-contact/local cues without requiring full socket visibility during final insertion.

## A1 Global target lock

Evaluate target identity on unobscured/wide-view frames.

Do not require target identity to be re-established every frame.

## A2 Local top-pin/top-hole observation

Create a local keypoint detector.

Start with deterministic geometry/edges if reliable.

If deterministic extraction remains brittle, train a small synthetic keypoint heatmap detector:

Input:
- RGB only.

Privileged training labels:
- top pin projected keypoint;
- top hole projected keypoint.

Privileged labels are evaluator/training targets only.

Network output:
- top pin heatmap;
- top hole heatmap;
- visibility/confidence.

Use domain randomization for:
- brightness;
- contrast;
- texture;
- mild viewpoint error;
- occlusion;
- motion blur.

Do not train the final residual yet.

## A3 Temporal tracking

Add a small temporal tracker such as:

- Kalman filter;
- optical flow;
- short history.

A temporary visual miss should increase uncertainty rather than immediately deleting target identity.

Record:

- direct observed keypoint;
- tracked estimate;
- uncertainty;
- visibility.

## A4 Perception acceptance

Report separately:

- target-lock accuracy;
- top-pin visibility;
- top-hole visibility;
- relative residual accuracy;
- occlusion recovery;
- UNRESOLVED rate.

Do not use per-frame RGB samples as independent trials.

---

# MVP-B — Proprioceptive Residual RL

This experiment is not blocked by MVP-A.

Goal:

test whether a residual policy trained only in simulation can recover a frozen nominal insertion under local perturbations using observations available on the real robot.

## B1 Frozen base

Use the already validated Frozen Nominal reference.

Do not modify it.

## B2 Training perturbations

Start only from local preinsert/align states.

Randomize:

1. plug-in-gripper Δx;
2. plug-in-gripper Δy;
3. plug-in-gripper small yaw;
4. socket small XY offset;
5. controller delay;
6. gripper state variation.

Do not add all physics randomization at once.

First phase:

geometry + in-hand pose only.

Use held-out perturbation amplitudes for testing.

## B3 Student observation

Residual Student may read only deployable observations:

History of approximately 0.5–1.0 seconds:

- joint positions;
- joint velocities;
- nominal action;
- command-feedback residual;
- gripper aperture;
- current phase.

Optional:

- top-pin/top-hole image residual;
- visual confidence.

Visual features must support dropout.

Residual must remain executable when vision is unavailable.

Forbidden Student inputs:

- exact plug pose;
- exact socket pose;
- exact contact pair;
- insertion depth;
- penetration;
- simulator task success.

## B4 Privileged training only

Privileged simulation state may be used for:

- reward;
- curriculum;
- success labels;
- contact-event labels;
- teacher diagnostics.

It must not enter the Student observation.

## B5 Residual policy

First learned policy:

small GRU or compact MLP/history encoder.

Do not use a large Transformer.

Output initially:

- Δx
- Δy

Optionally Δz only after XY recovery is validated.

Bound every residual action.

Final control:

`a_final = a_nominal + clip(delta_a)`

## B6 Training algorithm

Use PPO first unless the existing project already has a better tested continuous-control implementation.

Do not train the nominal base.

Only train the residual.

Reward components:

- positive insertion progress;
- large terminal success reward;
- penalty for penetration/collision;
- residual magnitude penalty;
- unnecessary-action penalty;
- optional correct-contact-sequence reward.

Keep sparse/structured reward versions as separate ablations.

## B7 Baselines

Compare:

B0 Frozen Nominal

B1 Nominal + deterministic local search

B2 Nominal + Phase-Gated Residual RL

B3 Nominal + History Residual RL with visual feature dropout

Do not require Event-Gated visual logic to validate basic residual viability.

Only after B3 passes should Event-Gated learned control be added.

## B8 Residual MVP GO condition

On held-out perturbations:

1. Frozen Nominal fails on a meaningful fraction.
2. Residual improves success by a predeclared margin.
3. Nominal-condition regression remains small.
4. Student privileged input count is zero.
5. Residual remains functional under RGB dropout.
6. Correction magnitude remains within bounds.
7. No LLM calls.
8. No Twin mutation.
9. No hardware execution.

Recommended first gate:

- at least +20 percentage-point recovery improvement over Frozen Nominal;
- or statistically supported improvement with bootstrap CI above zero;
- nominal-condition success degradation <=5 percentage points.

If this fails, stop before attempting real transfer.

---

# 7. Event system V2

Events are now divided into:

## Directly observable events

Examples:

- TARGET_LOCKED
- LOCAL_VISUAL_ALIGNMENT_AVAILABLE
- GRIPPER_STATE_CHANGED

## Inferred events

Examples:

- IN_HAND_POSE_DRIFT_LIKELY
- CONTACT_MODE_LIKELY

These may be learned from history.

## Privileged training/evaluation events

Examples:

- exact contact pair;
- exact plug-in-gripper transform drift;
- exact insertion depth.

Privileged events are allowed as training labels/rewards but not runtime inputs.

Do not require every physically meaningful event to be directly visually observable.

This is a key architectural change.

---

# 8. Event comparison semantics remain

Keep:

- MATCHED
- PENDING
- CONTRADICTED
- UNRESOLVED
- UNEXPECTED

Do not weaken UNKNOWN semantics.

Instead fix observation architecture.

UNKNOWN remains UNKNOWN.

The system solves persistent UNKNOWN by:

- observation reacquisition;
- modality switching;
- uncertainty-aware temporal estimation;

not by pretending UNKNOWN = FALSE or TRUE.

---

# 9. Fast/Slow integration

Fast path:

- Frozen Nominal;
- deployable observations;
- temporal state;
- residual;
- safety;
- event/status update.

Slow path:

only after a stopped/failed episode:

- inspect failure capsule;
- compare privileged simulation diagnostics;
- propose new perturbation family or structural Twin hypothesis.

No Slow LLM in the high-frequency controller.

Known residual correction must not trigger a Slow model call.

---

# 10. V2 reporting

Create:

`experiments/event_residual_mvp_v2/`

Required:

- `PERCEPTION_REPORT.md`
- `RESIDUAL_REPORT.md`
- `grasp_dynamics_audit.json`
- `observation_ablation.json`
- `residual_training.json`
- `heldout_results.json`
- `rgb_dropout_results.json`
- `source_manifest.json`
- `FINAL_V2_REPORT.md`

Report Perception and Residual results separately.

A perception NO-GO must no longer automatically prevent the proprioceptive residual experiment.

A Residual NO-GO does prevent later real deployment experiments.

---

# 11. GitHub publication

Run all relevant old/new tests.

Commit every effective stage.

Push final branch:

`research/event-residual-dual-v2-20261123`

Verify remote branch head matches local HEAD.

Do not overwrite or delete:

`research/event-residual-dual-20261123`

Its NO-GO is retained as V1 evidence.

Do not upload raw videos, simulation assets, XML or private hardware code.

Final report must explicitly state:

- Simulation-only;
- no real training;
- no real execution;
- no real success claim.
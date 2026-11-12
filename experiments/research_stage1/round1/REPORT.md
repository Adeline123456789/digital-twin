# Stage 1 · round 1 — phase-localized real/sim discrepancies

Status: **PASS for measured discrepancy understanding; physical root cause UNKNOWN**. No physical model was calibrated in this round. Baseline is cc17a89; the same historical executions were analyzed before and after the tool change.

## Method and source inspection

[ASPIRE §2.1](https://arxiv.org/html/2607.00272v1#S2.SS1) addresses ambiguous rollout failures by exposing primitive inputs, outputs, execution status and selected observations. Its [RunRecorder.span / _after_step](https://github.com/NVlabs/ASPIRE/blob/fa00f0ef2bc1d91d594b9aa502c22133c9f538f7/src/cap_harness/artifacts.py) records calls and physical steps separately. Its [Robosuite adapter](https://github.com/NVlabs/ASPIRE/blob/fa00f0ef2bc1d91d594b9aa502c22133c9f538f7/src/cap_harness/robosuite/adapter.py) keeps step acceptance distinct from benchmark success. We adapt this mechanism to existing phase logs; importing its simulator/provider stack is unnecessary and incompatible with the existing Piper interface. This is mechanism reuse, not a claim to run ASPIRE.

[Recova §3.2–3.4](https://arxiv.org/html/2610.01178v1#S3) separates task execution, recovery and subsequent physical verification. We read its method, but did not adopt its reconstruction/DAgger pipeline: no new hardware rollouts or trained recovery policy are available. No Recova code execution is claimed.

Initial source inventory: RAX main is an index and implementation lives on embodiment-specific branches. SEAR's current repository contains README/assets/license and lists code release as pending. Neither was misrepresented as an installed or evaluated implementation. They remain candidates for later stages, not adopted methods in this round.

## Experiment and results

`compare_replays.py` now reports per-phase motor errors, threshold-sensitive onset, and separately labeled simulation object/contact summaries. Two real-command replays and three formal simulation recordings were analyzed; these are five analysis reports, **zero new robot/task episodes**. Sixteen tests passed, including early-phase localization versus a larger abort-only error and excluding positive contact distance from active contact counts.

| Matched real episode | Old whole-episode gripper RMSE | Grasp RMSE | Insert RMSE | First sustained >2 / 3 / 5 mm discrepancy |
|---|---:|---:|---:|---|
| residual | 1.857 mm | 7.202 mm | 1.040 mm | grasp, 23.767 / 23.800 / 23.900 s |
| grip-hold | 6.036 mm | 8.347 mm | 0.441 mm | grasp, 23.667 / 23.733 / 23.833 s |

Onset requires five consecutive sampled values within one phase. These are diagnostic thresholds, not new success criteria or measured slip labels. Previous image-change candidates occurred during insertion; the motor mismatch is already present during grasp. This does not show that grasp mismatch caused the later image change.

At the grasp tail, real feedback is approximately 49.5 / 51.0 mm while the original simulation is about 39.9 mm despite commanded opening reaching 27 mm. Thus a transient-only explanation is insufficient for the sustained contact plateau. Free-space approach error is about 0.4 mm. Geometry/contact and opening-coordinate calibration are competing explanations; neither has been uniquely identified.

The two older real05 candidate recordings show plug–strip contacts already during align (23 / 22 sampled frames) and throughout almost all insertion frames. The mass-fit candidate still fails. **Their feedback was transformed by an unvalidated affine opening map** (scale 1.0173063, offset −9.1979 mm); their 1.293 / 0.786 mm errors must not be compared directly with native-width replay errors as a clean physics improvement. Raw provenance and parameter status remain in the source metadata.

## Decision and next round

Retain the observation-tool improvement. It localizes a reproducible motor discrepancy on two historical real executions and identifies a coordinate-mapping confound in prior fits. It does not identify real contact forces, exact object slip, or a calibrated physical parameter. The phase-1 gate is satisfied only at this stated evidence level.

Enter Stage 2 to test the identifiable part numerically: first separate actuator/observation timing from the sustained contact-width discrepancy, using one real execution for fitting and another for validation. Test a bounded timing parameter before touching geometry/friction/COM. Reject the candidate if it merely shifts timestamps, worsens held-out phases, or cannot improve the actual simulation predictions. Do not promote the old affine opening map without independent physical evidence.

## Cost and scope

Analysis wall times are in each result JSON. Zero planner calls/tokens, zero new simulation integration seconds, zero hardware commands, zero runtime human interventions. Research-agent token usage is unavailable (null), so total research cost is unknown. Task success, valid episodes/hour and tokens per valid success are **not evaluated** by this offline analysis. Motion, original speed, safety constraints, success detectors, raw recordings and source XML were unchanged.

Evidence: diagnosis reports 20261112T034222Z_b939c01a, _1b9a1e22, _2efc945e, _8dd88394, _28dbfd04 contain dataset-relative references, SHA256 and numerical summaries. Removed formal folders were not recreated.

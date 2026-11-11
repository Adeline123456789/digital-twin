# Executable Agent learning and continuation results — 2026-11-11

This round connects actual Luna planning to the existing physics program. The model returns bounded JSON; it cannot call CAN, shell commands or edit XML through this adapter. Successful and failed physical attempts are archived under the prescribed local routes. Nothing from the independent simulator source or raw media is uploaded.

## Actual A/B/C/D execution

On the original retained piper_simulation interface at held-out holder X +0.5 mm, A/B/C/D each chose offset 0 and physically succeeded on their first attempt. There is no evidence of improvement on this easy case.

| Arm | Physical task success | Input/output tokens | Loop wall seconds |
|---|---:|---:|---:|
| A | 1/1 | 20817 / 107 | 165.1 |
| B | 1/1 | 21322 / 124 | 149.0 |
| C | 1/1 | 21487 / 95 | 146.4 |
| D | 1/1 | 21615 / 98 | 143.0 |

Times include CLI, physics and publication overhead; pure inference latency remains unknown. All models had a two-call/two-execution ceiling, but success stopped them after one. No new real motion or runtime human intervention occurred.

## Unknown target-frame error and actual execution feedback

A synthetic target-frame bias was withheld from model prompts. After D's first physical failure, the adapter reported only the observed precontact object-to-hole residual in socket coordinates. It did not reveal the true injected bias. D chose a new action and physically ran it.

| Condition | Model action sequence (mm) | Outcome | Input/output tokens |
|---|---|---|---:|
| D: +2 mm hidden bias, holder +0.75 mm | 0 → −2.18 | failure → success | 42840 / 267 |
| D: −2 mm held-out bias, holder −0.75 mm | 0 → +1.82 | failure → success | 42584 / 244 |
| C: +2 mm bias, holder +0.75 mm | 0 → 0 | failure → failure | 21311 / 135 |
| A: +2 mm bias, holder +0.75 mm | 0 → 0 | failure → failure | 20474 / 97 |

D recovered 2/2 episodes within two executions each; A and C each recovered 0/1. D repeated no failed action; A/C repeated one. **The repetition is imposed by the declared frozen-plan comparator**, not proof that those LLMs would spontaneously repeat when allowed feedback. C-vs-D is exploratory: C was added after observing D, and was not a blinded confirmatory study. Some failed full trajectories also hit the unchanged penetration tolerance; they remain failures and are never used as valid successful demonstrations.

VS002 records the generic signed precontact-residual correction rule, with two opposite-sign validation episodes. It contains no precomputed answer for a new bias. This demonstrates learning an unknown step within one insertion task family under privileged simulator pose, not general competence on unseen robot task families or real vision. No statistically reliable cost advantage is established; feedback uses more LLM tokens. The ~20k input-token overhead per CLI call is a next optimization target.

## Real failure observation and twin evidence

Three historical wrist videos were decoded and aligned to recorded monotonic command time. Yellow-housing appearance-change candidates occurred in insert; three saturation thresholds differed by 1–5 frames. Local visual review confirmed the tracked component in the grip-hold episode is the plug. This is observation-tool improvement, not calibrated 3D slip/contact/force diagnosis. Unique real root causes remain UNKNOWN.

Retained historical twin metadata reports gripper RMSE 1.2932 mm versus 0.7856 mm for a later mass/shape candidate. Both physical tasks failed. Independent object-motion validation and identified force data are absent; the candidate was not accepted simply for fitting motor feedback. No six-axis force measurements were read or invented.

## Final source discovery and adapter evolution

A newer final scene was found at dataset/calibration/offline_workspace_20261110. Intermediate sysid directories were deleted, but final source, portable camera config and the collection entry remain. This corrects the earlier overbroad implication that final source could not be located. Its source hashes and results are separate from the original interface.

The documented final-scene command succeeded unchanged. Connecting it to the learning adapter then exposed a software integration error: the adapter inherited the old 12°/s free-space timing, although the retained documentation prescribes 25°/s (existing contact cap still 12°/s). At holder +0.25 mm, two model-selected insertion offsets both failed transport IK. These failures were saved; changing insertion offset did not solve that run.

After restoring the documented profile, the same +0.25 mm initial state and zero offset succeeded. Independent −0.25 mm initial state also succeeded. Only the free-space timing forwarding changed between the matched failed/successful zero-offset trials. The software mismatch is verified; the detailed physical chain through grasp transform/IK remains model-specific. No controller, friction, mass, XML, safety limit or real hardware setting was changed.

The adapter now supports the retained profile and model selection through environment variables; nine unit tests pass, including explicit forwarding of the documented speed/grasp/alignment arguments. Original and retained-source success denominators must remain separate. Version logs preserve candidate/previous evidence and the active policy is not promoted on narrow real-world claims.

## Remaining boundaries

No real movement was authorized or attempted. No new physical calibration was accepted: independent real pose validation is missing and the existing failure status explicitly reports unmodelled flip/timestep sensitivity. The permitted workflow branch is recorded rejection/deferment, not invented success. There is no permanent unattended hardware/self-modification service. Explicit task batches can use the new model-to-existing-simulator entry.

New robot task-family success, physical force identification, real deployment and total cost reduction remain unproven. Current verified progress is a bounded failure/feedback/repair loop, a reusable simulation skill, better observation evidence, and an independently checked adapter correction.

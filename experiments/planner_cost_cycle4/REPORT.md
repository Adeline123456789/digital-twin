# Cycle 4 — reject the general planner replacement

Four actual calls and four complete simulator Episodes: three successes and one failure. No new real movement. The preregistered promotion condition failed; the default remains `codex_default`.

| Requested model | Profile | Input / cached / output tokens | Call seconds | Valid success | Planning + collection seconds |
|---|---|---:|---:|---:|---:|
| gpt-6-luna | codex_default | 16748 / 7168 / 207 | 23.29 | 1/1 | 195.65 |
| gpt-6-luna | bounded_json_v1 | 13338 / 7168 / 152 | 24.14 | 0/1 | 146.41 |
| gpt-6-astra | bounded_json_v1 | 13344 / 7168 / 194 | 24.48 | 1/1 | 195.57 |
| gpt-6-astra | codex_default | 17379 / 7168 / 177 | 26.80 | 1/1 | 197.08 |

Luna input decreased 20.4%, but its structured action was `offset_mm=2` while the reason proposed1. The actual2mm command failed stable insertion and subsequently crossed the unchanged penetration limit. The spoken repair was not executed. These facts support rejecting this candidate in this batch, not a universal causal claim about all shortened instructions.

Astra input decreased23.2% and both profiles succeeded. Each group has only one sample; this does not establish reliable latency or general ability improvement. The same7168cached input tokens were reported in all four calls. The four calls used61539input+output tokens in total, including the failed candidate.

Pooling the two model cases descriptively, baseline produced2/2valid Episodes with17255.5planner tokens and196.36planning+collection seconds per valid success. Candidate produced1/2with27028tokens and341.98seconds per valid success. Failed attempts must stay in the numerator. Research/orchestrator usage, financial pricing and hardware costs are not included; total collection cost reduction remains unproven.

Every successful motion lasted54.4simulated seconds. Maximum commanded joint speed was11.99593deg/s in all four. Successful-run maximum sampled command acceleration was54.05–54.06deg/s²; the failed run reached100.39deg/s² and stopped at41.533s. Its shorter duration is a failure, not improved throughput. Native task, physics, safety thresholds, original smoothing and nominal12deg/s speed were unchanged.

All three successes are under `complete_sim/succeed/<episode_id>/`; the failure is under `complete_sim/failed/20261116T050715Z_a6d1c3df/`, with complete per-Episode videos/state/outcome and existing verification. `result.json` maps all parent and physical IDs. Simulator privileged object/contact/task information was used: Simulation-only.

The candidate stays explicitly opt-in for diagnosed experiments and is not promoted. Return to the unchanged default by omitting `ROBOT_PLANNER_PROFILE` or setting it to `codex_default`. Shared memory records the command/description mismatch; future teacher feedback must address the emitted numeric action and real execution evidence. No teacher/student or held-out exploration result is claimed by this cycle.

Tests: three planner contract tests and one shared-reference comparison test passed. Paired real comparison code added for cycle3 is prepared but has only synthetic validation; it has not manufactured a real result. Cycle3 resumes after fresh hardware checks; cycles5–6 remain available within the original six-cycle cap.

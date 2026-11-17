> Historical offline checkpoint below. Superseded by [the authorized batch and real comparison](BATCH_REPORT.md); its former pending-authorization statements describe the earlier checkpoint.

# Goal-aligned continuation — offline tool checkpoint, 2026-11-17

Parent: `63e835f767e7b5d6cd72651955fcf1fa74e7d6b1`; branch: `codex/goal-aligned-runtime-20261117`.
The new mission is development guidance, never a runtime prompt. Existing 12 cycles remain consumed. The previous 15-trial hardware batch is not resumed. This checkpoint uses already recorded simulation evidence, zero new physical/simulated Episodes and zero inner-model API calls. Research/orchestrator token cost is not available per checkpoint and is **not zero**.

## Actual state and changes

`agent_loop.py` still defaults to `llm`. Existing `diagnose_then_repair` already supported retention/IK diagnosis, offset, screened tilt and stop. This continuation adds an **opt-in** `bounded_tools_v1` route in that same entry: model JSON requests dispatch to Episode summaries, bounded trace windows, contact inspection and existing repair diagnosis, then the unchanged complete simulator collector handles an accepted plan. No arbitrary native tools or hardware are authorized through this interface; native model tool events cause rejection. The underlying Codex backend remains read-only, and its startup/context cost has not disappeared.

The candidate preserves recorded grasp height, alignment gap and initial joints when changing one parameter. Height/gap proposals use the existing adapter bounds; no new physical parameter search, motion slowing, smoothing, safety change or success-detector change. Missing model usage remains unknown, including failed/timed-out calls. The planner deadline bounds model calls; synchronous diagnostic tools stop cooperatively between calls, so this is not a certified hard real-time bound.

SQLite caches hash only relevant files when inode/size/mtime/ctime change. Analysis keys include source content, analyzer, extraction parameters and applicable runtime versions; contact keys additionally cover archived model and referenced mesh content. Raw videos remain local and are never decoded by these tools. Cache storage is private under `.local/`. Old→new→old content invalidation is unit-tested, **not a completed scene-transfer experiment**. `record()` whole-source snapshot took about 0.010 s; it was not replaced because the measured cost does not justify more machinery.

`collect_inline.py` adds a MuJoCo integration-state field to future privileged rows, using the installed `mj_getState` interface. Existing fields/readers and raw recordings are preserved. This includes state needed to distinguish actual applied forces/equality activation from defaults. Only unit round-trip verification has run; the next authorized complete collection must verify this extension with the native dataset checker.

## Archived evidence experiment

| Source Episode | Existing physical success | Summary cold s | Summary reuse s | Contact cold s | Contact reuse s |
|---|---:|---:|---:|---:|---:|
| 20261116T114608Z_4d3972ba | True | 0.8429 | 0.0013 | 15.2294 | 0.0052 |
| 20261116T125354Z_b5546f36 | False | 0.8045 | 0.0013 | 0.8637 | 0.0037 |
| 20261116T125834Z_58df2069 | True | 0.6163 | 0.0014 | 0.8234 | 0.0035 |

All three cached outputs exactly match their uncached JSON results. Reuse hashed zero source bytes. Cold contact costs share process/import/mesh cache effects; they are not independent cold-process trials. Summaries are 4.6–7.7 kB from 6.1–16.2 MB state/contact traces, not measured model tokens. These are **offline analysis latency measurements**, not successful Episodes/hour, total collection cost or LLM capability improvements.

The first run `20261117T045930Z_66f8146a` failed its cold/warm equality check: contact-pair tuples became JSON arrays on cache reload. The failure report is retained. The fix uses JSON-stable arrays and has a regression test. The subsequent run `20261117T050031Z_8b21b868` passed all three archived cases. This is one engineering continuation with one failed check and its repair, not two newly allocated research cycles.

## Contact evidence and limits

The selected grasp frame in each Episode exposes contact distance, solver activation, effective inclusion margin, geom gap/margin, condim/friction, force estimates, visual-mesh distance, equality constraints, applied-force state and actuator limits. Historical recordings omit full integration state and independent original mesh hashes. Therefore these results are `RECONSTRUCTED_PARTIAL_STATE`, using archived XML plus currently referenced meshes, and are **not recorded forces**. Historical finger width is split equally between the two fingers during reconstruction. External force injection, earlier direct object edits and callbacks remain UNKNOWN; default zero arrays do not rule them out.

The inspected successful 20 mm grasp has active finger contacts close to visible mesh surfaces. This does not reject a visual gap elsewhere or identify the cause of real slippage. Simulator prismatic force in N has no identified conversion to the real gripper motor torque in N·m. No real six-axis force or depth observations were introduced. No digital-twin calibration has been accepted.

## Targeted prior work read

- [REFLECT paper, §3.1–3.2](https://proceedings.mlr.press/v229/liu23g/liu23g.pdf) and [actual SceneGraph source](https://github.com/real-stanford/reflect/blob/main/main/scene_graph.py), plus `LLM/prompt.py`: failure explanation through hierarchical subgoal/event evidence. We reuse the coarse-to-fine evidence mechanism with existing traces; we do not install its AI2THOR/RGB-D/point-cloud stack or its unbounded retry loop. Current real observations remain RGB-only. This is a mechanism adaptation, not a reproduction of REFLECT benchmark results.
- [MuJoCo 3.3.7 contact reference](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#body-geom-gap): a recorded candidate contact can be inactive; read `efc_address`, effective margin and solved force separately. The installed Python API is used directly. Mesh collision is based on convex geometry; nearest visible mesh distance alone is not a full geometric validation.
- [MimicGen actual pose-transform source](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/utils/pose_utils.py) and [data-generation API](https://mimicgen.github.io/docs/modules/datagen.html) were inspected. Object-relative end-effector targets require downstream planning; they are not joint-array translation. No code was copied and no scene adaptation is claimed in this checkpoint.
- [RHD full method §3](https://arxiv.org/html/2609.33378v1): teacher guidance is revised from recipient execution feedback. Its VLA interventions are not the current Piper controller. The next study must use actual student tool decisions, teacher feedback, recipient re-execution, frozen held-out evaluation and full costs. No RHD implementation or teacher/student result is claimed here.

## Verification and promotion

35 targeted tests passed (after updating two fake-tool fixtures to include the actual cache-key contract), including an actual `agent_loop.run` routing test with a stub model/collector, unknown-tool rejection, unscreened-tilt rejection, numeric-action preservation, integration-state round-trip, inactive-gap contacts, timeout usage and cache invalidation. Stub-model checks are engineering tests, not LLM experiments. Native hardware source, MuJoCo source/assets, independent success criteria, motion limits and default active policy are unchanged. Rollback is selecting the existing `llm` policy; no candidate is promoted.

| Required capability | Status now | Missing evidence |
|---|---|---|
| Inherit current repository and actual entry | PASS | Latest parent and route verified |
| Lower latency runtime | PARTIAL | Cached offline analysis verified; actual model/collection cost pending |
| Reusable contact diagnosis | PARTIAL | Three archived states; historical forces and full-scene cause unknown |
| Local scene update and old→new→old | NOT IMPLEMENTED in continuation | Actual state/trajectory adaptation and regression |
| Real–Sim reference pipeline | PARTIAL / existing | Preserved; no newly authorized real execution |
| Strong teacher → weak recipient feedback | BLOCKED on new experiment budget | Actual calls and recipient executions |
| Teacher-free held-out exploration | BLOCKED on new experiment budget | Frozen tasks/tools/memory and independent physical sim results |
| Effective self-evolution/default use | PARTIAL | Candidate wired; fair comparison, promotion and later use not yet run |

Engineering: **CONDITIONAL GO for the offline diagnostic candidate**. Research/full goal: **NO-GO for a completion or cost-reduction claim**. Overall goal remains active. Do not treat this report as completion.

## Proposed next bounded batch, not yet authorized

Reserve at most 16 actual model calls, an observed 250k input+output token stop threshold (cached tokens are part of input, not added again), at most 8 complete simulated Episodes and 90 minutes. No real movement or resets. A final in-flight call may cross an observed token threshold; record the overrun, do not silently reset the budget. Use current available model endpoints only; no training or downloads. First freeze current source/task/tool/memory versions and verify a compatible development seed; then run recipient-first development, one teacher feedback revision, paired recipient comparison and a teacher-OFF held-out layout test. Stop the direction if no information is gained; do not spend all capacity by default. If physical-sim capacity is exhausted, teacher/generalization results remain incomplete rather than deleted. This proposal does not grant those resources or a new hardware batch.


## Prepared recipient experiment runner (same continuation, no new budget consumed)

`scripts/benchmark_recipient_feedback.py` now prepares and executes the specified development/teacher/recipient/held-out sequence when explicitly launched with an authorized design. `recipient_design.json` is **PROPOSED_BUDGET_PENDING**. Its source Episode was checked against the current native source and task hash. No model call or new simulation was started while preparing it.

The proposed study uses one existing source scene, a development offset perturbation and a distinct held-out holder layout. These are same-task parameter/layout conditions, not unseen task families. Models receive no held-out answers. Parent and candidate share tools, per-run empty analysis caches, identical task observations and a three-call planning ceiling. Teacher guidance is frozen before held-out construction; no teacher call exists in the held-out execution path. The planned sequence uses at most 13 calls and six full simulated Episodes, leaving two simulation slots inside the requested ceiling for separately recorded independent checks. The parent is generic guidance under the same new tool harness; this is **not yet** a legacy-default versus candidate-harness performance comparison.

Budget accounting records every returned/failed model call, actual input+output use, simulations and wall time. Missing usage stops further spending; cached input is not double-counted. Failed recordings remain failures and stop the study. A stopped planner is not counted as physical success. Full teacher/development costs remain in the ledger. No automatic default promotion occurs.

39 targeted tests now pass, including a stubbed full study-order test, failed-call ledger handling, teacher-context exclusion and no-plan accounting. A test fixture initially supplied no budget object and was corrected to the real interface; this was not a model or physical experiment. These checks establish code contracts only. Actual teaching, independent exploration, scene-transfer improvement and successful-Episode cost remain unverified. The budget question is pending; elapsed time is not approval.

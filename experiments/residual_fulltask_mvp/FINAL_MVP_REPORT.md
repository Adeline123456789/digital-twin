# Full-Episode Post-Grasp Residual MVP Final Report

## Decision

**Stage A: NO-GO. Stop before Stage B.**

Failure category: `HELDOUT_GENERALIZATION_FAILURE`.

The five-seed pure-proprioception History PPO experiment completed, but B2 improved over Frozen Nominal by only **11.875 percentage points**, below the preregistered **15 percentage-point** minimum. The other Stage A conditions passing does not override this mandatory effect-size gate. No Stage B–F work was started.

## Frozen Stage A execution

- Candidate: 15-frame proprioceptive History PPO with structured reward; no RGB or privileged channels.
- Training seeds: 51001–51005, exactly 16,384 policy steps and 200 rollouts per seed.
- Total training: 81,920 policy steps, 1,000 rollouts, five fixed-budget final checkpoints.
- Evaluation: 64 paired held-out conditions and 20 paired nominal conditions for B0, B1, and each B2 seed; 588 rollouts total.
- Statistics: paired hierarchical bootstrap over training seeds and test conditions.

| Variant | Held-out result | Change vs B0 | 95% CI | Nominal |
| --- | ---: | ---: | ---: | ---: |
| B0 Frozen Nominal | 9/64 (14.0625%) | — | — | 20/20 |
| B1 Deterministic Local Search | 20/64 (31.25%) | +17.1875 pp | [6.25, 28.125] pp | 20/20 |
| B2 History PPO, mean over 5 seeds | 83/320 (25.9375%) | **+11.875 pp** | **[0.625, 23.4375] pp** | 100/100 |

B2 held-out successes by training seed were 8, 19, 17, 19, and 20 out of 64. Four of five seeds improved over B0, with per-seed changes of -1.5625, +15.625, +12.5, +15.625, and +17.1875 percentage points. Thus the confidence, seed-stability, and nominal-regression criteria passed, but the mean effect-size criterion failed.

B2 was **5.3125 percentage points below B1** on average, with 95% CI [-16.25, 4.375] pp. The interval crosses zero, so this experiment neither establishes a difference nor supports a claim that PPO is better than deterministic recovery.

## Interface and safety audit

- 588 evaluated actions passed bounds and left-arm invariance checks.
- Maximum residual on one Cartesian axis: 0.0011038287 m.
- Maximum residual-axis speed: 0.0012788480 m/s.
- Maximum joint correction: 0.0066376647 rad.
- Privileged Student Inputs: 0.
- Hardware executions, real-data reads, real training, runtime LLM calls, automatic Twin mutations: all 0.

Validation before the experiment passed 7 focused tests. The compatible repository suite passed 181 tests plus 27 subtests; one test that would import the prohibited private-control root remained excluded. A local nominal runtime witness had exact integration-state error 0 and zero privileged inputs.

## Required research questions

1. **V2 是否可复现？ — NO-GO.** The implementation and fixed-budget rollout protocol reproduced operationally, but the five-seed result did not reproduce an effect large enough for the V3 Stage A gate: +11.875 pp versus the required +15 pp.
2. **完整 Nominal 是否可重放？ — INCONCLUSIVE.** Stage B was not permitted after Stage A NO-GO. The local Stage A nominal witness is not the required full Episode audit.
3. **抓取后的位姿偏差是否能够被观测？ — INCONCLUSIVE.** Stage C was not run.
4. **Proprio 是否足够？ — INCONCLUSIVE.** Stage A shows a positive but insufficient local signal; it does not answer full-task observability.
5. **是否需要右腕 RGB？ — INCONCLUSIVE.** No RGB observability comparison was permitted.
6. **Residual 能否在 LIFT / TRANSPORT 阶段恢复？ — INCONCLUSIVE.** Full-task perturbation and recovery were not run.
7. **Residual 能否继续完成 INSERT？ — INCONCLUSIVE.** Full-task continuation was not run.
8. **Full-Task Residual 是否优于只在插入阶段修正？ — INCONCLUSIVE.** No Full-Task Residual was trained.
9. **PPO 是否优于确定性方法？ — NO-GO.** B2 trailed B1 by 5.3125 pp and its paired interval crossed zero; superiority is unsupported.
10. **当前仍有哪些 Sim-to-Real Gap 没有解决？ — INCONCLUSIVE.** Perception/extrinsics, contact and friction mismatch, actuator/gripper dynamics, latency, in-hand pose drift, and real recovery behavior all remain unresolved. This simulation-only round makes no real-transfer claim.

## Evidence and review status

The machine-readable decision and statistics are in `reproduction.json`; exact source, result, and checkpoint hashes are in `source_manifest.json`.

The evidence precheck verified all seven submitted result claims. The selected independent DeepSeek result-to-claim review returned no complete response because its remote connection closed during a chunked transfer. Per review policy it was not automatically retried; semantic-review verdict is `REVIEW_UNAVAILABLE`. This does not soften or replace the preregistered deterministic `NO-GO`.

## Stop record

The taskbook explicitly requires Stage A NO-GO to stop before Full-Task PPO. Therefore the later-stage environment, observation, training, evaluation, and runner modules were intentionally not created. No placeholder modules or speculative fallbacks were added.

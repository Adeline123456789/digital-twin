# CONTACT_MEDIATED_DRIFT_V1 Residual Report

**Finding: NOT_SUPPORTED.**

The core experiment used the frozen full Episode, unchanged physics, a contact-mediated lift disturbance, and the deployable P0 history only. Requested disturbance direction and privileged drift were not Student inputs.

## Frozen challenge

The distribution was frozen before training: lift-middle, 8 N for 50 ms, with angle uniform over [[-0.28, -0.268], [1.05, 1.08]]. The formal pilot had 13/23 Frozen-Nominal successes and no rejected condition inside the final distribution.

## Paired held-out results

| Method | Success | Improvement vs B0 | Paired 95% CI | Recovered | Spoiled | Collisions |
|---|---:|---:|---:|---:|---:|---:|
| B0 Frozen Nominal | 35/64 | — | — | — | — | 0 |
| B1 Deterministic Search | 35/64 | 0.00 pp | [0.00, 0.00] pp | 0 | 0 | 0 |
| B2 PPO seed 62001 | 35/64 | 0.00 pp | [0.00, 0.00] pp | 0 | 0 | 9 |
| B2 PPO seed 62002 | 28/64 | -10.94 pp | [-18.75, -4.69] pp | 0 | 7 | 14 |
| B2 PPO seed 62003 | 0/64 | -54.69 pp | [-67.19, -42.19] pp | 0 | 35 | 23 |
| B2 PPO mean | 32.81% | -21.88 pp | [-51.56, 0.00] pp | 0.00 | 14.00 | 46 total |

## Integrity and interpretation

All 3 policies are fixed-budget final checkpoints. Paired-condition integrity: True. Action bounds: True. Privileged Student inputs: 0. Hardware executions: 0. Held-out tuning: 0.

The support label follows the amendment directly: strong support requires a consistent positive effect with a positive paired interval; limited support permits a positive but seed-sensitive effect; no reliable improvement is not supported. No arbitrary percentage-point threshold was imposed.

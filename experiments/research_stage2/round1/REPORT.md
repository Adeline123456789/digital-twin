# Stage 2 · round 1 — timing hypothesis tested, no physics promotion

Experiment: `20261112T034638Z_193f29d6`. Outcome: **timing-only hypothesis insufficient; Stage 2 gate NOT PASSED**.

## Reused numerical method

The official [MuJoCo sysid optimizer](https://github.com/google-deepmind/mujoco/blob/71bc51ce2194ddcdf876de5ecd05bdf57a75e93b/python/mujoco/sysid/_src/optimize.py) uses bounded least-squares, including a SciPy backend. Its [SignalTransform.apply](https://github.com/google-deepmind/mujoco/blob/71bc51ce2194ddcdf876de5ecd05bdf57a75e93b/python/mujoco/sysid/_src/signal_transform.py) windows data before resampling predicted measurements with a delay. These are measurement parameters, not physics parameters. We inspected both implementations. Local MuJoCo is 3.3.7, without `mujoco.sysid`; SciPy 1.15.3 is already installed. We reused `scipy.optimize.least_squares` directly for this one-parameter residual instead of upgrading the user's simulator or porting a toolbox.

The positive delay bound is 0–300 ms: an exploratory envelope for effective response/logging lag, not a measured CAN bound. Training uses only residual-episode grasp commands between 55 and 59.5 mm, before the observed plateau. The independently recorded grip-hold episode is held out from fitting. Both share the planned task; this is episode-level validation, not unseen motion-family validation. All candidates use the same valid time window to prevent extrapolation or changing the denominator. No timing, gain or bias in the actual robot or simulator was changed.

## Results

Five optimizer evaluations, fitted effective delay **45.622 ms**. Analysis took 0.085 s; no new LLM planner calls or physics integrations.

| Observation window | Baseline RMSE | Candidate RMSE |
|---|---:|---:|
| Training precontact | 0.4043 mm | 0.0650 mm |
| Held-out precontact | 0.4253 mm | 0.0778 mm |
| Held-out full grasp | 8.2846 mm | 8.1607 mm |
| Held-out insert | 0.4135 mm | 0.4148 mm |
| Training abort-retreat | 0.7071 mm | 1.1839 mm |

Metrics here are evaluated on original real command timestamps, unlike round 1's camera-rate grid; do not treat the denominator difference as improvement. The fitted lag explains some free-closing transient error but neither the sustained contact-width discrepancy nor later failure. It also worsens some phases. It mixes plant response, observation timing and the old 30 Hz replay discretization, so it is not an identified hardware delay.

**Decision:** preserve the numerical result as a limited measurement-model observation; reject global deployment and physical-calibration promotion. No physical candidate becomes active; rollback is therefore unnecessary. Eighteen tests pass, including synthetic delay recovery and a constant contact mismatch that no delay can remove. Tests are software checks, not evidence of real calibration.

Next experiment follows the remaining evidence: test whether one stable opening-coordinate correction explains both grasp and later held-object phases on the independent episode. If it cannot, stop treating a low fitted grasp error as evidence that the opening map or contact model is calibrated. Further physical fitting requires independently observed aperture/object pose, not just a renamed motor residual.

# Stage 2 · round 2 — reject a phase-specific opening fit as calibration

Experiment: `20261112T034933Z_271a9075`. Stage gate: **NOT PASSED**. This round follows the timing test: its remaining sustained contact error motivated an opening-coordinate hypothesis, not another prompt change.

Reuse: SciPy bounded least-squares and the same loaded real/simulation pairs. MuJoCo's [SignalTransform.bias](https://github.com/google-deepmind/mujoco/blob/71bc51ce2194ddcdf876de5ecd05bdf57a75e93b/python/mujoco/sysid/_src/signal_transform.py) explicitly represents additive measurement corrections separately from physical parameters. Only one offset was fitted, using the final 0.5 s of residual-episode grasp. Bounds ±12 mm cover the historical, unvalidated ~9 mm opening correction; they are exploratory bounds, not certified physical tolerances. Timing and all physical parameters stayed fixed. The other real episode and other phases were never used for fitting.

Fitted offset: **+9.6122 mm to the predicted opening**, nine evaluations, 0.083 s CPU analysis. A constant offset gives an excellent training plateau fit and an apparently good held-out plateau improvement, but fails its cross-phase prediction:

| Window | Baseline RMSE | Candidate RMSE |
|---|---:|---:|
| Training grasp plateau | 9.6123 mm | 0.0196 mm |
| Held-out grasp plateau | 11.1104 mm | 1.4983 mm |
| Held-out approach | 0.4046 mm | 10.0154 mm |
| Held-out transport | 0.6868 mm | 10.0751 mm |
| Held-out insert | 0.4135 mm | 9.8222 mm |

Reject the candidate. It cannot be a globally applicable constant opening correction for the original model. This does not disprove all nonlinear/load-dependent mappings or the different retained model's mapping; neither is identified here. The real encoder reading drops substantially during lift while the old simulation changes much less. Object seating, contact location and rotation are still confounded with physical aperture calibration. Encoder readings alone cannot identify friction, mass or COM.

No active model, original source or data was modified. Nineteen tests pass. Zero new task rollouts, planner calls or hardware motion; research-agent tokens remain unknown. No efficiency/success claim follows from these analysis-only experiments.

Next: use the existing raw RGB-D around grasp/lift to check whether independently measured fingertip/object geometry is available. Do not optimize contact physics against another scalar motor residual unless that measurement ambiguity is resolved. We will not fill the four-round budget with additional offset/lag searches.

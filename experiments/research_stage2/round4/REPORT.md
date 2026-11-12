# Stage 2 · round 4 — independent camera/table validation rejects promotion

Experiment: `20261112T040059Z_e90f39df`. **Candidate rejected; Stage 2 NOT PASSED after four rounds.** No Stage 3/4 experiments are authorized by the current stage gate.

## Reuse and physical meaning

Read the actual [Fischler–Bolles RANSAC paper, §II](https://graphics.stanford.edu/courses/cs233-21-spring/ReferencedPapers/ransac.pdf): fit minimal subsets, count compatible observations, then refine the consensus. We directly invoked the existing `piper_simulation/scripts/fit_plug_table_depth.py` without copying or modifying it. It deprojects recorded aligned depth with the RealSense SDK's declared distortion model, fits a table plane by RANSAC, and refines with SVD. Its source SHA256 is in the result. No new numerical optimizer or dependency was introduced.

One plane constrains camera height and two normal directions. World XY/yaw and table world z=25 mm remain existing CAD/scene assumptions. This is a scene-observation calibration candidate, not aperture/contact/dynamics identification. The existing fitter produces a candidate; the wrapper rejects it if normal change exceeds 2 degrees or height change exceeds 20 mm, rather than clipping it or changing hardware limits.

Training: residual episode frames 0/30/60. Held out: grip-hold and smooth episodes, frames 0/30/60, same camera identity verified locally. Both old and new transforms are evaluated on identical points in fixed existing table ROIs with baseline-plane distance below 20 mm. No candidate-specific validation mask is used. Acceptance was fixed before fitting: bounds pass, actual MuJoCo loading passes, and both held-out episode RMSE values improve at least 10% with nonincreasing p95 error. The 10% is an engineering retention threshold, not statistical significance.

## Results

| Recording | Baseline plane RMSE | Candidate RMSE | Reduction |
|---|---:|---:|---:|
| Residual (train) | 3.4783 mm | 3.4617 mm | 0.48% |
| Grip-hold (held out) | 3.5576 mm | 3.4415 mm | 3.26% |
| Smooth (held out) | 3.3869 mm | 3.3280 mm | 1.74% |

Candidate normal change: 0.0968 degrees; camera height change: +0.3491 mm. Bounds passed. The candidate was loaded through the **actual existing MujocoBackend camera-calibration interface**; model camera position/orientation matched the candidate JSON. The default scene/configuration was not changed. No task rollout was run.

Both held-out changes are below the predeclared retention threshold. Existing depth biases and correlated pixels remain; roughly 145k depth samples per recording are **not 145k independent experiments**. Two separate recordings validate only this station/sensor, not external metrology or cross-installation calibration. No claim is made that these small plane improvements explain the 7–8 mm grasp discrepancy.

Decision: reject promotion, retain baseline and preserve candidate locally at `calibration/20261112T040059Z_e90f39df/`. No XML, safety limit, motion speed or success criterion changed. No active rollback was necessary because the candidate was never installed. The comparison and old/new numerical transforms are public; raw depth, private camera metadata and source remain local.

## Failures and cost

An initial script startup had a mismatched closing bracket and failed before fitting or loading any model. `startup_error.json` preserves it; it was corrected before the recorded experiment. No successful fit was fabricated for that failure. The recorded fit/validation/load check took 10.363 s. Twenty-three tests passed. Planner calls/tokens = 0; per-round research/orchestrator usage was not separately metered, so total cost remains unknown.

## Stage decision

Four adaptive rounds tested timing, aperture mapping, observation validity and an observable scene-calibration candidate. None established an accepted, independently validated physical-twin improvement. Therefore Stage 3 collection benchmarks and Stage 4 self-evolution comparisons remain **NOT RUN — prerequisite unmet**. Do not spend more rounds changing friction/COM against unverified object/aperture observations. The next prerequisite is validated cross-camera/object/fingertip correspondence and geometry, preferably using existing recorded material; new robot movement remains prohibited.

# Cycle6: matched spatial observation confirms registration disagreement

The user's correction redirected this final system-improvement cycle from model benchmarking to real/sim spatial investigation. One open-gripper observation batch completed, with no grasp or insertion attempted. Both proposed hand-eye fits were rejected. No camera, scene, contact or dynamics parameter was promoted, and no physical centimetre error is claimed. This is an unsuccessful calibration investigation with useful measured disagreement, not a successful digital-twin repair.

## Physical execution and data

Reference `3707fab7f1638a526bea5e4d86e9be477f5cfbf43e75f11055ec33e95ebab482` passed the unchanged0.2mm penetration gate at all physics substeps, plus a stricter no-robot/task-contact check for this observation purpose. Simulation privileged contacts, poses and rendering were used. The first simulated design was superseded before hardware; the second included wrist rotations and a15mm X +15mm Z translation at each region. Simulated pose limits prevented a more vertical high vantage, so the validated path used a reachable tilted stand-off pose. No joint limit was relaxed.

Real execution sent all9,053 reference joint/gripper targets in exactly the same order and values, using the existing right-arm controller and supported-shutdown lifecycle. All23 sampled holds had zero encoder change within the100ms window around the selected RGB exposure; maximum target/feedback joint difference was0.204deg. This checks stationary feedback, not external kinematic accuracy. No contact was intended; no measured force or depth is available to certify its physical absence.

**Target arrays match; timing does not match exactly.** Nominal reference90.52s, actual command span101.369s (+11.985%). No dynamic equivalence or preserved real execution time is claimed. This timing issue did not produce the measured stationary-view image disagreement and remains a separate collection limitation.

Both raw RGB streams have3,122 frames. The existing synchronized exporter produced3,104 frames in each video and the right-only HDF5; complete movement/command coverage and the native verifier passed. The complete observation Episode is:

`self_evolution/spatial_cycle6/episodes/20261116_real_spatial_probe_000001/`

It contains `videos/`, `proprio_stats/`, `raw/task/`, `outcome.json`, `pair_manifest.json`, and `verification.json`. It uses the existing `connector_mixed.v2` format and `wrist_alignment_probe` purpose, with `physical_task_success=null` and `policy_training_eligible=false`. It is excluded from successful/failed insertion datasets and task success-rate denominators. Compatibility links retain existing reader paths. No raw data was uploaded.

After the operator confirmed support, the existing controller disabled the arm and exited. A separate read-only check confirmed all six motors disabled, err_code0. There is no active motion controller. Task-motion corrections0; shutdown-support events1; operator confirmation messages1. Execution LLM calls/tokens0; research-orchestrator usage is separate and not assumed free.

## Measured appearance disagreement

At corresponding stationary holder/socket views, simple component measurements give the following real-minus-sim displacement at640×480. Gray thresholds30/40/50 check sensitivity for the first two features; they are not confidence intervals or physical measurements.

| Visible feature | Horizontal pixels | Vertical pixels |
|---|---:|---:|
| Holder black rectangular feature, median | −2.68 | +81.19 |
| Yellow front-hole dark region, median | +6.82 | +63.56 |
| Upper visible blue socket panel | −0.20 | +29.47 |
| Lower visible blue socket panel | −0.85 | +26.27 |

The yellow-hole vertical result varies62.15–63.79px over the three thresholds. These centroids measure appearance, not calibrated3D target centers. Slight tracked joint differences and CAD/rendered shape differences remain. The observations establish visible disagreement; they do not establish a single rigid object offset or its size in centimetres. See `image_disparity.json` and the locally referenced paired images.

The current scene/camera initializer is explicitly provisional and not qualified for hardware motion. Its historical RGB-D label does not make its depth a real observation. The active right-wrist transform was checked against the initializer and is identical. Controller link6 pose agreeing with nominal URDF FK (prior two attempts:0.099/0.127mm RMS) is a software consistency check, not proof that the camera, object or physical TCP is calibrated.

## Two bounded calibration candidates, neither accepted

The [method note](METHOD.md) documents the actual paper and source read. The implementation uses installed OpenCV feature processing and SciPy least-squares for SHECAR's fixed-FK/fixed-intrinsics reprojection formulation. No GPL source was copied, no framework was installed, and no simulator XML was changed. Five distinct views per region enter fitting; combined-angle views and return views are excluded. Initializer, bounds, sampling and gates were saved before fitting.

| Candidate | Training landmarks | Reserved-view result | Decision |
|---|---:|---|---|
| SIFT + mutual matching |16, socket only | Socket median7.30→1.86px; no holder coverage | Reject: cannot establish the required two-region result |
| Shi-Tomasi/LK + forward/backward check |24, both regions | Holder5.45→2.18px; socket7.38→6.04px;100 evaluation limit reached | Reject: no convergence and thresholds failed |

The second candidate changed only the correspondence method; its reserved frames had already been inspected for candidate1 and are not called independent acquisition evidence. Neither candidate was tuned after these outcomes. The roughly22mm mount changes are **unvalidated fitted parameters**, not measured object displacement. Their directions disagree.

An additional static-render audit used the fixed finger regions, excluded from the feature-fitting mask. It compared simulated privileged finger segmentation with real dark-finger masks at three gray thresholds. Candidate1 made the finger overlap much worse (IoU0.478→0.160 at threshold60). Candidate2's small apparent improvement was not stable across thresholds (threshold80:0.516→0.495). This test also depends on actual CAD fidelity, so it does not isolate camera error from finger-geometry error. It independently reinforces the rejection of a camera-only correction. No extra fit, physics run or robot movement was used for this audit.

## Retained changes, limits and next useful work

Retained: a hash-checked partial observation tool using the original controller, RGB/FK correspondence extraction, a tested offline registration evaluator, stationary-view image measurements, and a rule to cross-check scene fits against robot-attached geometry before deployment. Four targeted unittest cases passed, including exact synthetic metric-transform recovery, optical-flow displacement, open-gripper speed/fold and reference rejection. The native dataset verifier passed after archival through its compatibility paths.

Rejected: both fitted camera transforms; no new real Skill, success-rate improvement, physics calibration or lower total cost is claimed. The original active planner remains `llm`/`codex_default`, and the ordinary simulated grasp height remains20mm. The12mm real repair stays rejected. This added tool did not change hardware safety or successful-task detection.

Budget used:2/3 physical sim probe validations,1/2 noncontact real batches,2/2 candidate fits,0 full task retries,0 execution-model calls. A second physical batch was not consumed merely to repeat rejected candidates. The six-system-cycle batch is now evaluated with unmet goals retained; the scientific objective is not achieved. The next high-information intervention is a known-size RGB target or verified robot-attached geometric landmarks with wider, independently validated observation motions. It must distinguish camera mounting, physical fingertip geometry and scene placement before any further grasp-height/friction repair. Repeating the same untextured small-orbit fitting or a blind full insertion is not supported by these results.

Evidence: `real_result.json`, `fit_001.json`, `fit_002.json`, `fit_design.json`, `fit_002_design.json`, `hold_stationarity.json`, `image_disparity.json`, `mount_visual_audit.json`. All successful observations and rejected candidates are preserved.

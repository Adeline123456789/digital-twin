# Stage 2 · round 3 — audit independent geometry evidence

Stage gate: **NOT PASSED**. The failed timing and aperture fits leave a need for independently observed object/fingertip geometry. We checked existing three-camera RGB-D at grasp/lift transitions; no new acquisition or hardware motion.

## Observed failures and repair

1. `20261112T035409Z_12a40111`: the analysis ran, but global largest-yellow selection produced a misleading external component at a constant ~0.48 m and tiny unrelated left-wrist components. Visual inspection of the external and left-wrist views showed that these were not reliable target correspondences. Those results are retained and **excluded from geometric calibration**.
2. `20261112T035542Z_5743eacf`: ROI update failed with `too many values to unpack (expected 4)` because a finger-mask temporary overwrote the ROI tuple. This infrastructure failure is preserved. The temporary was renamed; no fallback or new dependency was added.
3. `20261112T035706Z_ed39796e`: repaired run completed, 16 samples across two historical episodes and two applicable cameras, 6.976 s. Target ROIs were selected from the training episode's grasp/lift images and frozen for the other episode. Left wrist was excluded because the inspected view did not contain the target. These limited ROIs are not a general object detector.

The design follows the trace-selective inspection already motivated by ASPIRE, using existing OpenCV segmentation and the camera's recorded exposure clocks. It uses no external learned perception model and fits no physical parameter. The new test checks a larger yellow distractor outside the target ROI; missing depth remains unknown rather than zero distance.

## Results and limits

Right-wrist target depth coverage is only **0.78–7.47%** in these selected frames, with some far-background contamination. It cannot be treated as a dense, independently calibrated grasp-surface point cloud. Finger rectangles also contain background; their medians are not physical aperture measurements.

The corrected external target has **75.8–88.1%** nonzero depth over 141–180 eroded mask pixels. Median depth is ~0.730 m at grasp end and ~0.684–0.686 m at lift end across the two recordings. This is useful measured evidence about visible surfaces in the camera frame. It is not object-center pose, calibrated gripper aperture, or a force measurement. Occlusion and changing visible surfaces prevent treating centroid displacement as rigid-body ground truth without further validation.

The result changes the next experiment: usable external RGB-D exists, so declaring all real geometry unavailable would be wrong. Before fitting contact from it, evaluate the existing camera/table registration against separate real episodes. The retained and original extrinsics are explicitly provisional; pose errors can otherwise be absorbed into contact parameters.

## Next and budget

Stage 2 has one research round remaining. Reuse the simulator's existing `fit_plug_table_depth.py` (RANSAC followed by SVD), fitting only the observable table normal/distance on one historical episode, and test on two other historical recordings. Keep CAD-anchored XY/yaw and all contact parameters fixed. Only accept a bounded improvement supported by both independent recordings; otherwise report the stage gate failure and do not enter Stage 3.

No formal episodes were generated or deleted. Original simulation source, XML, safety limits, speed and success detector remain unchanged. All three runs, including the failed update and invalid global correspondences, are saved. Planner tokens/calls = 0; research/orchestration tokens are unknown. Motion quality and data-collection efficiency were not evaluated here.

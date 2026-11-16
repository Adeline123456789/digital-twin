# Cycle 3 checkpoint — shared reference validated; real task pending

The current right-only native simulation includes the physical folded-to-approach
motion and completes the task. It produced a complete successful Episode at
`complete_sim/succeed/20261116T044726Z_6d665f3f/`: two actual configured simulation
RGB channels, 1,632 frames, 54.4 s simulated motion. Collection plus verification
took 171.975 s. The Episode is not nested in another Episode.

The frozen 100 Hz controller reference has 5,438 points and duration 54.3667 s.
Its independent MuJoCo replay passed the original task success detector and
every integrator-step penetration check: maximum 0.03104 mm against the unchanged
0.2 mm numerical bound. Validation took 116.002 s wall time. No friction,
geometry, contact parameter, motion speed or smoothing change was used.
The source native rollout's maximum penetration was 0.03146 mm.

This is **Simulation-only**, using privileged object/contact state. The reference
contains simulation commands, not simulated measured joint trajectories. Its
immutable hash and prediction trace hash are in `reference_validation.json`.
Passing both simulations does not establish real alignment, physical calibration,
or a reduction in collection cost.

## Real readiness and recording

The existing hardware controller's read-only preview passed with only the right
arm connected. Measured joints were `[0, -2.380, 1.155, 0.476, 22.254, 2.232]`
degrees. The unchanged bounded folded-entry rule maps J2/J3 to zero; this entry
matches the simulated first command to less than 0.000009 degree. No enable or
motion command was sent. Fresh external and wrist RGB images show the task
objects and no person in the visible work area; this observation expires and
must be refreshed for actual execution.

The native RGB/CAN recorder was exercised without motion. Its right-only export
contains 14 common frames in both videos and a matching six-joint HDF5 stream,
with no fabricated left-arm channels or depth measurement. The hardware Python
environment lacked `av`; export succeeded using the existing simulator Python
environment. The initial diagnostic export retained an obsolete RGB-D purpose
label, which was corrected and re-exported. Both diagnostic outputs remain
outside the formal Episode roots. See `recorder_check.json` for evidence.

The local runbook explicitly requires supporting the arm before disabling it,
and documents J5 falling after disable. Current operator availability for that
physical condition has been requested. This is not a renewed request for normal
task authorization. Until it is confirmed, the robot stays disabled; no controller
or recorder service has been left running. The cycle remains open, not failed or
complete. A fresh readiness check is required after a response.

## Implementation and checks

Public additions: named reference adapter, physics replay validator, three small
contract tests and reviewed evidence. Native task/capture/export interfaces were
adapted locally for the right-only RGB workflow; source remains private, with
backups and hashes in `local_integration.json`. The actual execution entry loads
the frozen reference and checks its matching successful physical validation.
The old hardware safety lifecycle is unchanged.

Contract tests: 3 passed. Existing agent integration tests: 6 passed. The strict
reference adapter and native legacy resampler were also compared on the prior
right-only success and produced exactly equal commands, times and phases; the
strict adapter additionally rejects trajectories requiring time stretching.

No real transfer denominator exists yet: 0 real task attempts, 0 demonstrated
real successes, no calibration and no claim of weak-model independent discovery.
Planner calls in this cycle: 0. Research/orchestration model cost is separate and
not yet fully accounted. There has been one operator-availability question and
no physical intervention during this cycle. The six-cycle budget remains intact:
cycles 1–2 completed, cycle 3 pending real execution; cycles 4–6 not spent here.

# Cycle 3 — shared reference reached hardware; grasp retention failed

**Real task failed, complete recording retained, robot safely disabled.** This
cycle establishes a concrete reference/command/feedback pair; it does not establish
successful transfer or calibrated contact physics. The earlier pending checkpoint
is retained in Git history and `reference_validation.json`.

| Trial | Observed result |
|---|---|
| Native simulation including folded entry | Successful,1,632frames,54.4s; complete Episode |
| Frozen100Hz reference in MuJoCo | Successful,5,438points; all original substep checks passed |
| Real right-arm task |0/1success;3,137identical reference-prefix commands, then existing abort/retreat/fold |

Reference ID: `dd8950e77027d0106eb260240db235b91aea8e59b808ad6b23e54d706f244b11`.
The real controller used this reference directly. No online LLM generated new
hardware actions. Its unchanged wrist RGB protection stopped before insertion
because the plug was absent. Recovery is preserved and not counted as completion
of the reference. The actual prefix lasted31.981s versus31.36s nominal; the0.621s
accumulated logging/scheduling lag remains visible, with no time warping.

## Earliest supported discrepancy

The important discrepancy is during **grasp/lift**, before the final align-stage
alarm. At grasp end, commanded aperture was27mm and motor-reported aperture38.2mm.
By20.948s after the first reference command, during lift, feedback was within1mm
of the empty commanded aperture for five consecutive samples; lift end was27.1mm.
RGB frames650,670,710 show the plug remaining at the holder while fingers withdraw.
The simulated aperture instead remained about39.96mm at grasp end,39.46mm after
lift and38.44mm after transport. Six-joint command tracking RMS was below0.36deg
per joint in the examined phases; this is not an independent TCP calibration.

This supports **loss/non-establishment of a retained grasp during lift**. It does
not uniquely identify friction, grasp height, lateral registration, geometry or
available gripping effort. The first tracking-threshold crossing during grasp is
compatible with intended object contact and is not itself a failure detector.
The controller label `insertion slip` identifies where its guard fired, not the
physical root cause. Root cause remains UNKNOWN. Motor aperture/effort is not a
six-axis force measurement; RGB does not provide a verified3D object trajectory.

## Recording and safety outcome

Complete real Episode:
`complete_real/failed/20261116_real_shared_reference_000001/`.
Both actual camera videos contain1,590decoded frames (53s), with matching right
HDF5 channels, true failed outcome, verification and pair manifest. Raw RGB/CAN,
sent commands and execution report are together under `raw/task/`. The former
staging capture path is only a compatibility symlink to that same directory.
There is no left-arm stream, real depth, force sensor, or fabricated successful
label. Maximum camera pairing skew was14.58ms; recorded movement and commands
are fully covered. Failure remains excluded from successful-policy training.

The native verifier still assumed two arms; its minimal local adaptation reads
actual arms and cameras from existing v2metadata. It passed this1,590-frame real
Episode and the existing1,133-frame dual simulation regression. Original source
stays local; backups and hashes are in `local_integration.json`. The generic raw
capture summary still calls its tool purpose calibration; authoritative export
metadata explicitly records `yellow_pick_insert_attempt` and preserves that source
label without reclassifying the task as a calibration experiment.

User confirmed operator availability and subsequently actual support. After the
native fold completed, the existing `DISABLE` lifecycle exited normally. Independent
read-only feedback confirmed all six motors and gripper disabled, err_code0 and
software-stop arm_status1. J5 then relaxed to23.026deg; this is recorded separately
from the enabled folded zero. No background hardware controller remains.

## Costs, versions and next decision

No planner model was invoked for this real task. Research orchestration cost is
separate and not yet attributable by cycle. Physical task interventions0;
operator support for shutdown1, with two condition-confirmation exchanges. Real
success denominator is zero, so cost per valid real success is undefined.
Simulation collection/verification took171.975s, adapted-reference validation
116.002s, and the recorded real attempt53s including preparation and recovery.
These are components, not total elapsed or financial cost.

Named adapter tests3, agent tests6 and paired-comparison test1 passed; actual
comparison additionally verified all3,137sent prefix values. The interface and
recording are validated on this failed attempt; no successful real skill or
physical parameter calibration is promoted. Version and raw evidence references
are in `real_result.json`; source rollback restores the backed-up native scripts
without removing the current right-only hardware restriction.

Cycle5 should discriminate the grasp/contact relationship using these real
observations, update only supported twin state/model or operation, validate the
candidate in MuJoCo, then return to the authorized task. It must not respond to
this failure by merely correcting insertion offset. Cycle4's separate cost study
is already complete; the batch remains within the six-cycle total cap.

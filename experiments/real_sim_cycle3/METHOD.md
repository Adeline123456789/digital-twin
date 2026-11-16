# Cycle 3: shared executable reference

The concrete gap was that the earlier simulation initialized at the grasp approach,
while the disabled physical right arm was folded. A successful insertion rollout
did not validate that entry motion. The existing task now receives the measured
entry through its existing `--start-right-deg` interface. No model/contact parameter
was calibrated in this cycle.

The existing native task, recorder, exporter and Piper controller are reused.
The public addition freezes named simulation **commands**, converts radians to
degrees and gripper width to the controller's existing 95 mm normalization, and
interpolates on the original timeline at 100 Hz. It rejects excessive speed rather
than stretching time. The resulting commands undergo a second physical replay
with the existing integrator, contact bound and independent task detector before
hardware execution. Reference, simulated state, sent commands and real feedback
remain separate evidence.

## Targeted primary-source review

| Source and material actually read | Original mechanism | Decision for this system |
|---|---|---|
| [D-World](https://arxiv.org/html/2610.17114), methods 4.1–4.5 | Same-control replay residuals; state correction, parameter revision and reconstruction distinguished; candidate non-regression | Adopt the evidence separation and replay test. No official implementation was established. Paper depth observations are not available on this RGB-only robot. One mismatch cannot identify friction or exact fixture pose. |
| [CoEnv / CoEx v2](https://arxiv.org/html/2604.05484v2), methods 3.2–3.3; [actual replay source](https://github.com/MARS-EAI/CoEnv/blob/5a232525f5b1845ffe038ac96573fec811def2da/coenv/replay_actions.py) | Privileged execution, executable program reuse and predeployment trajectory/collision checking. `ActionReplay` uses its Gymnasium environment and semantic controller actions. | The existing local replay lifecycle already supplies the relevant hardware interface. Do not install a second framework or claim its simulation replay class directly controls this Piper. Reviewed MIT source at commit `5a232525f5b1845ffe038ac96573fec811def2da`; no third-party source copied. |
| [Real-is-Sim](https://arxiv.org/html/2504.03597), methods 3.1–3.2 | Real follows simulated actual joint state; image-driven state correction with fixed physical parameters | Useful distinction between state synchronization and model learning. This cycle instead freezes simulated reference commands. No Gaussian rendering stack or online visual action generator adopted. |

These works motivate a narrow experiment; they are not evidence that our robot
has been calibrated or that collection cost decreased. The numerical contact
profile remains unidentified from real measurements. RGB-only recording and
right-only channel mapping were checked independently before task execution.

## Budget and interpretation

See `design.json`: cycle 3 of six total, at most two full native simulations,
two adapted-reference validations and one real task. A completed command sequence
is not physical success. Safety abort, unknown outcome, recorder failure and
direct-reference success must retain separate labels. A recovery or manual
correction cannot count as successful unmodified reference transfer.

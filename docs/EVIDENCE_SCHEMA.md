# Evidence contract

The schema is defined by the dataclasses in `robot_agent/contracts.py`. A measurement binds an episode, native source SHA-256, source profile, world configuration, recorded policy identity, task and outlet, phase, action, frame, units, source modality and clock. The recorded policy identity hashes the recorded adapter digest and action arguments. It identifies what was recorded. The source/config scope must be reviewed again for another scene.

| Measurement | Frame and unit | Evidence |
|---|---|---|
| `sim.privileged_contact` / `sim.object_pose` | world, m | Recorded physics or an explicitly identified reconstructed frame |
| `sim.socket_residual` | socket, mm | Existing native precontact transform |
| `sim.native_report` | simulator, status | Terminal report; last sample time can be a lower bound |
| `real.rgb` | image, px | RGB-derived estimates require uncertainty |
| `real.gripper` / `real.proprio` | gripper mm / joint deg | Recorded motor feedback with a known clock |

Each observation records observed value, expected value, residual and uncertainty when available. A reference names its root (repository or dataset), relative path and SHA-256. The Episode reader cross-checks source and verification records, decoded-frame manifest, hashed RGB files, privileged frames/poses, and H5 simulation time. A missing input, wrong frame, unit or unknown provenance yields an unavailable observation. `FailureReport.task_success` preserves the original native result independently of these checks.

Events carry their observation plus kind, severity, status, first observed frame/time, confirmation time, hypothesis IDs and root-cause status. Three consecutive carry samples must satisfy the frozen no-contact and extra 10 mm separation predicate. This is the existing 30 Hz reference method. A timestamp gap resets trust in the stream. Earlier onset between samples is unknown. The terminal IK exception in the development case has a lower bound of 18.300 s and a null exact observation time.

The relative clock for simulation is H5 `sim_time`. Recorder receipt and command CSV monotonic clocks have different meanings. The stopped-capture reader uses the existing camera-to-host mapping and returns the native stop, encoder measurements and the explicit prior label. Real depth and six-axis force values are null. Its command times have no independent CAN send-time measurement.

The failure capsule projects at most three events, three hypotheses and five reviewed memory references. Physical evidence, suggestions and scoped decisions retain distinct labels. Revoked real-depth knowledge and unverified narrative contribute no executable skills. The only executable memory in this replay adapter is the historical dual-arm VS002 profile. Other unresolved entries carry a rejection reason in the retrieval result.

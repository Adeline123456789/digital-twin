# Frozen grasp comparison

Both diagnostics replayed the same reference and initial state, without video or hardware. The original condim6 lifted 52.91mm; condim3 lifted65.47mm but changed object position/orientation. Neither reproduced the observed real grasp loss. Effective contact dimensions and sliding friction0.7 were checked in contact records. This does not support changing the active contact model.

The original20mm center height contacts a sloping shoulder (normal vertical component0.371). Candidate14mm reduced that to0.215;12mm reached the cylindrical rim (approximately horizontal normals). Both native prefixes lifted within the unchanged0.2mm penetration limit. This is privileged simulated geometry and an operation hypothesis; no physical calibration or real improvement is claimed. Duplicate contacts across mesh tetrahedra are not independent contact evidence.

The optional diagnostic stopping flag is local and does not change default full execution. The additive collector now passes and records an explicit bounded grasp-height candidate, retaining the original20mm default. A full recorded candidate and an independent initial condition remain to be checked.

Friction-cone self-check, collector boundary/default/candidate check, three reference adapter tests and two right-arm integration tests passed. pytest is absent from the native runtime; the function self-check and existing unittest runner were used without installing dependencies.

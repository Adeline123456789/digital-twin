# Digital twin discrepancy assessment — round 1

No physics/XML parameters were changed or accepted in this round. Existing calibrated scene is provisional; its source SHA-256 is recorded in every experiment. Trial changes are deliberate target offsets and held-out holder initial positions, not system identification.

Historical evidence: four real task failures vs existing simulation successes; two repeated 805-row real plans are exactly equal. Real execution reports and command CSV hashes are in diagnosis reports. Real gripper feedback measures motor position, not object pose or contact force. Existing notes report slip/contact ambiguity, unqualified hand-eye transforms and a prior visible/collision mesh mismatch subsequently repaired. These statements do not identify a unique real physical cause.

Measured new difference: 2 mm target bias fails in the current MuJoCo model; removal succeeds at nominal and ±1 mm holder initial positions. This is model sensitivity, not a real calibration residual. No new real joint/EE/object trajectory was measured. Target logs and model trajectories cannot substitute for independently measured object motion. Consequently joint/EE/object RMSE, friction error and real execution-time alignment remain unavailable in this round.

Next discriminating experiment: independently observe plug-to-gripper pose before transport/contact and pin-hole alignment; pair timestamped real feedback with the exact final command timeline in the existing replay script. Keep camera and object observation uncertainty explicit. Only then select a parameter family (fixture transform, grasp transform or contact friction), define physically measured bounds, fit on designated episodes and validate on different episodes. Existing ±50 mm hand-eye boundary-hitting candidates must not be silently accepted. No optimizer was run because reliable paired physical targets and held-out real measurements are missing.

Status: CALIBRATION_DEFERRED. Hardware access remains unused; no new human permission for movement was requested or assumed.

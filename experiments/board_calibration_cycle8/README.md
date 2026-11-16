# Cycle 8 preparation: existing known-size target

The user confirms a physical calibration board is available. Its type, row/column convention and metric spacing remain required inputs; an asynchronous question is pending. Do not infer dimensions or dictionary IDs.

Existing OpenCV5.0.0 exposes chessboard-SB and aruco; six existing hand-eye/pose tests passed. Reuse those interfaces after the actual target is known. No new solver, dependencies, calibration fit or robot motion has been introduced. This is preparation, not an eighth completed experiment.

Use RGB target coordinates in explicit metres, exposure-matched joint feedback and a rigid stationary target. Keep withheld configurations out of fitting and report closure separately from independent metric validation. Store device transforms only in the local dataset. Planning/capture must continue to use the established right-arm controls and safety checks.

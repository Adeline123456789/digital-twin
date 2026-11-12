# Stage3 round2 — candidate not yet promoted

The first paired native/inline run has identical physical trajectories, numeric HDF5 data, privileged JSON and task outcome. Collection component time is 249.2143 s versus 150.5236 s (39.60% lower in one pair). The preregistered exact decoded-pixel criterion failed, so the benchmark stopped before failure/heldout cases. This is not an accepted efficiency improvement.

The unmodified baseline also fails pixel-exact repeatability. Two short native probes show raw RGB differences before encoding, so the difference cannot be attributed solely to H264 or the inline adapter. Implementation-level rendering cause remains UNKNOWN. See first_comparison.json and render_repeatability.json.

Simulation-only. Simulator object/goal poses, joint/action state, contacts, task success, camera poses and qvel were used. No native controller, speed, physics, independent success rule or safety rule changed. Stage2 real calibration remains unvalidated.

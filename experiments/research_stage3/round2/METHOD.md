# Stage 3 round 2: avoid duplicate integration

Round 1 measured 90–105 s for a physics trial, followed by 131–145 s for a second physics run that records videos. Inspecting `try_plug_pick_sim.main`, `collect_plug_sim_episode.main`, `checked_physics_step`, `state_row` and `MixedEpisodeWriter` confirmed this duplication.

## Prior implementation and applicability

[robosuite paper, sections 2 and 2.5](https://arxiv.org/html/2009.12293v3) separates task/control execution from sensor observations. Its [DataCollectionWrapper source](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/data_collection_wrapper.py), particularly `step`, records state/actions after native execution; reset/flush manage episode boundaries. We read the implementation, including its XML/state reset used for deterministic replay. The original problem is repeatable demonstration capture for learning, not proving faster collection on this Piper system.

Direct import would require a robosuite environment and change the existing dataset format. Instead, the new adapter reuses the local native physics checker, state-row builder, camera renderer, writer and archiver. It observes each completed native physics interval; no robot control or solver is replaced. Unlike robosuite's reset workaround, this adapter does not reload or modify XML. Performance is an experimental hypothesis.

The current task script has no observer argument. Two imported symbols are rebound only within an isolated invocation and restored afterward: backend construction enables cameras, and the native physics-step function is called before recording. Phase labels are attached from the original final trace. This narrow integration depends on those existing interfaces; source hashes are checked. Success uses the original task result, all native collision checks, and native stable seating truth. The format verifier does not decide task success.

A recording compared with its own run proves alignment, not independent deterministic replay. Promotion additionally requires paired native replay versus inline runs with identical full trajectories, final states, numeric HDF5 fields (except host clocks), outcomes and decoded RGB frames. All source/model/assets remain in the separate simulator and local dataset.

No new physics calibration, planner capability, generalization, production-throughput or total research-cost claim is supported by this experiment. Simulation privileged information is used as listed in design.json.

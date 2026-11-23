# Stage 0: current native interface

Scope: simulation only, dual-arm model with the existing right-arm task. Baseline is `87bd88181836bed5c4723bc7a79a2586060d032f`. The new research branch does not change either historical branch. Source hashes are in `source_manifest.json`. Graph MCP tools are unavailable and this checkout has no `.codegraph/`; findings below come from the named source files, not an asserted complete graph index.

## Native task and execution

- `collect_current_scene.py` selects `original` and `dual`, then calls `benchmark_collection.run_case` → `learn.simulate` → `collect_inline.collect` → the external `try_plug_pick_sim.main`. `--local-only` suppresses automatic Git publishing. No legacy planner/Skill entry is called on this path. `learn.record` hashes historical repository memory; this is warning/provenance access only.
- `MujocoBackend.__init__` resolves the existing local asset, composes the task scene, and loads a dual model when `active_arms` is `[left, right]`. It does not select `right_model`. Model, assets, task camera calibration and raw recording remain local.
- The action vector is `[left.joint1..6, left.gripper_width, right.joint1..6, right.gripper_width]`: 14 values, joints in radians and full gripper width in metres. Left indices are 0–6; right joints 7–12 and gripper 13. `sim.apply(sim.layout.command(...))` writes native position actuator targets. Recorded H5 joint angles are degrees; raw trial actions are SI.
- The native generator initializes both arms, then changes only right joints and right gripper. The left target remains zero joints / 0.060 m width. This is not coordinated dual-arm manipulation.
- Relevant bodies: `task/yellow_plug`, `task/board`, `task/strip`, `right/gripper_center`; finger joints are `right/joint7` and `right/joint8`. Target outlet comes from the unmodified task configuration (currently 3).
- With the requested smooth-timing/default start, phases are descend → grasp → lift → transport → align → insert → release → retreat → fold. Optional explicit start adds approach; non-smooth timing adds settle. The simulator starts at an IK approach pose, not a verified physical robot start.
- `place_over_socket` re-reads current gripper/plug/strip transforms and calls `PoseIK.solve` at transport, align and insert. It is a privileged nominal generator, not a frozen policy. The historical collector field `planner_calls=0` counts no language-model planner; it must not be reused as a claim of zero native IK replanning.
- There is no native Cartesian residual hook. The real simulation injection point is the per-frame joint command before `checked_physics_step`. A minimal frozen player can bypass `main`/`PoseIK`, replay recorded commands through `apply`, and retain checked native tracking physics. A later Cartesian residual must use robot-only kinematics and deployable observations; a policy must never receive the backend or `MjData`.
- `task/strip` is a static fixture: no joint/mocap is created for it. `reset_task` places it via model body transforms. A transient perturbation would need an explicitly evaluated runtime fixture-motion interface that moves collision and rendered geometry together. No such perturbation is executed at Stage 0.
- `connector_details.strip_geometry` constructs slot polygons by buffering actual pin polygons with `slot_clearance_m` (currently 0.0008 m). This is available for later amplitude design, not a calibrated real clearance. No pilot amplitudes are frozen yet.
- `configure_rigid_collection` supplies the existing numerical collection profile. The baseline uses it unchanged, including every-integrator-step 0.2 mm penetration rejection. This profile is not an identified real contact model. Native stable seating and complete-task checks remain unchanged.
- `collect_inline` records three RGB videos and all physics intervals; `finalize_recordings` supplements commands/initial state, checks schema/units, decoded frame counts and trace consistency, then installs a single complete Episode. It uses the historical **simulation** failure as the format reference. This is not access to real failure data.

## Existing Fast/Slow boundaries

The inspected modules are `contracts`, `diagnosis`, `first_divergence`, `fast_loop`, `supervisor`, `memory_router`, `local_slow`, and `gate`, together with both architecture/evidence documents.

`Reference` already binds root, relative path and SHA-256. `Observation.issue` enforces identity, version, clock, sensor, frame and units. These checks are reusable; the current modality list will need an additive deployable-simulation observation contract if Stage 3 is reached.

`EventBuilder` consumes stopped full Episodes and simulator privileged contacts/poses. It is suitable for separate evaluator evidence, not the Student input. `FastLoop` currently applies a privileged retention predicate. `Supervisor` defaults to zero model calls, but its default `MemoryRouter` can propose VS002. Therefore the MVP must not simply instantiate this default routing path as its live controller. A scoped router must deny legacy execution, while retaining existing evidence, failure capsule and supervision semantics.

`local_slow.local_diagnosis` calls a loopback model and is prohibited during this batch. `EvidenceGate` reviews pinned evidence; it does not write actuator or Twin state. Its existing narrowly scoped acceptance must not be interpreted as approval of the new residual. `agent_loop.py --arm D` is not the experiment entry.

## Regression scope

The old controller-import test `RightAgentTests.test_connection_rejects_absent_arm_before_controller_import` is excluded because it reads `RIGHT_ARM_CONTROL_ROOT`, explicitly forbidden for this MVP. The exclusion is a validation limitation, not a pass. Other old tests use local/synthetic evidence or simulation; runtime model calls in unit tests are mocked. Test tooling is installed only under ignored `.local/event_residual_mvp/testdeps`, leaving the MuJoCo environment unchanged.

## Stage decision

Proceed to the one requested zero-offset native baseline. Only a native success plus complete Episode verification permits export and physical replay of a frozen reference. A failed prerequisite is a NO-GO stop, not permission to change grasp, friction, thresholds, reference scene, or invoke a historical Skill.

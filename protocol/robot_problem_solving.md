# Robot problem solving protocol v2

Use scripts/agent_loop.py --arm D in the existing MuJoCo Python environment for complete recorded offline collection. scripts/learn.py remains the low-level simulation adapter; headless success is not a complete Episode. Keep existing robot APIs and safety limits.

1. Specify physical completion independently of API acceptance. Record task, version, initial state, action, expected and observed physical effects, earliest anomaly and final outcome. Missing observations remain UNKNOWN.
2. Retrieve at most five memory records matching task, failure signature and applicability. Read evidence references, not whole videos or chat histories.
3. Separate observations from hypotheses. Propose at most three causes, each with a falsifiable, low-cost experiment. A model explanation alone is not verification. Distinguish the first observed anomaly from the true onset if sampling is incomplete.
4. Compare the proposed repair and action fingerprint with negative experiences. Do not repeat an ineffective repair under the same conditions without new evidence. A repair after the earliest anomaly requires an explicit causal explanation.
5. Change one variable in the existing MuJoCo interface. Save config/source hashes, inputs, full local evidence references, metrics, failures and exceptions. Do not tune friction or delete collisions just to pass.
6. Replay original condition and predeclared held-out initial states. Promote only within the validated scope; record regressions and restore the previous active version on failure.
7. REAL_VERIFIED requires physical success evidence, simulation validation, existing safety checks and fresh human permission. SIMULATION_VERIFIED never authorizes hardware.
8. Persist JSON + Markdown for every attempt, including timeout and infrastructure failures. Run publish.py with explicit paths after each completed round; keep local commits on push failure.

## UNKNOWN TASK MODE / Exploration Protocol

State the goal and observable completion; split into subgoals; retrieve available skills with applicability; explicitly list unknown transitions. Generate <=3 candidate plans and predict each observable outcome. Prefer reversible simulator probes that separate hypotheses under equal budgets. Observe, update or reject hypotheses, and stop repeating contradicted plans. If evidence is insufficient, retain UNKNOWN and select the next discriminating observation. Promote a reusable skill only after independent initial-state validation. Never substitute a memorized unseen-task answer for this process.

## Metrics

Report task success, unseen-task success, repeated-failure rate, new skills, tokens, inference wall time, simulated time, actual robot time and interventions separately. Missing measurements are null, not zero. Deterministic software-policy tests are not weak-LLM capability evaluations. Calibration requires bounded parameters, real measured targets and held-out real data; otherwise defer it.

## Simulation privileged information

Simulator object/goal poses, contact state, velocities and task ground truth may be used in every simulation stage. Each result must disclose the fields actually used and mark conclusions Simulation-only. Model comparisons must expose the same information to each group, or explicitly separate observation access from method effects. Privileged access does not authorize changing success detectors, physical parameters or hardware safety rules. Count a successful collection Episode only after its existing-format videos, state/action arrays and physics replay are verified; headless success alone is insufficient.

## Executable memory and current scope

Agent D reads evolution/active.json.planner_policy. compiled_alignment_v1 may execute VS002 only when its simulator/source, socket, units, contact and initial-state checks match; otherwise use the bounded LLM path and keep unknowns explicit. Record actual_planner_kind and model_invoked. Retention loss observed during transport takes precedence over a later IK exception when selecting what to investigate; do not infer friction or measured force. The native success and safety detectors stay independent. All full collection passes the existing Episode verifier, including failures. Current evidence does not establish unknown-task-family competence, real calibration or total-cost reduction.

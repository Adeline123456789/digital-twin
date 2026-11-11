# Completion audit (active work, not a completion claim)

| Requested requirement | Current authoritative evidence | Status / next required work |
|---|---|---|
| Preserve existing systems; inspect Git/interfaces | PROJECT_AUDIT.md, source hashes, original checkout retained | Done for inspected paths; external orchestrator not fully located |
| 3–5 real failure cases | Four diagnosis result.json and source hashes | Done; unique physical root causes UNKNOWN |
| Separate actions from physical success | learn.py, execution_report imports, paired physical trials | Done |
| Earliest anomaly and <=3 hypotheses | Historical reports plus timestamp-aligned wrist feature reports | Improved; physical onset/causality remains unverified |
| Existing MuJoCo hypotheses and single-variable repair | Six physical offset trials and replay verification | Done in limited scene; not proof of real cause |
| Robot validation | No motion; no fresh hardware permission/readiness | Deferred as explicitly allowed, no REAL_VERIFIED claim |
| Five memory categories with provenance/status | memory/ | Present; useful skill discovery beyond outcome-check rule remains pending |
| Actual model execution feedback loop | scripts/agent_loop.py and round 2 preregistration | Running; inspect physical results before claims |
| Version comparison and regression rollback | evolution/, tests, deterministic v1/v2 evaluation | Implemented; not yet proven general physical improvement |
| UNKNOWN TASK exploration beyond memorized answers | Protocol and unknown routing | Implemented interface; new-task exploration demonstration still pending |
| Weak model A/B/C/D | Round 1 reasoning-only; round 2 executable runs | Physical trial batch in progress; same-family holdout is not unseen task-family proof |
| Tokens/time/interventions and failures uploaded | Per-experiment reports and Git commits | Continuing; missing historical metrics remain null |
| Sim-real calibration with bounds and held-out validation | Retained metadata audit, calibration gate, TASK_STATUS/cleanup hashes | Not accepted: missing independent pose/force data and removed candidate sources |
| Explicit data routes and no simulator upload | AGENTS.md, dataset indexes, original-video hashes, Git tracked-file audit | Done; raw artifacts remain local |
| Per-round GitHub publication incl failure | publish.py and verified remote commits | Ongoing; inspect latest remote before final report |

Calibration is deliberately not reported as complete. The retained historical mass candidate lowered gripper RMSE but did not reproduce full physical failure, lacks independent validation and has removed source artifacts. These are blocked evidence requirements, not reasons to invent a fit. Continue all software/observation/exploration work that does not require new hardware movement.

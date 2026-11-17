# Completed model response lost at the process deadline

Batch 2 call 3 recorded one completed agent message and `turn.completed` (19,408 input and 111 output tokens), but the CLI process exceeded its timeout. The shared receiver discarded every timeout result. The old log did not retain the answer text: its physical usefulness is UNKNOWN and the historical student outcome remains unchanged.

`model_compare.ask` now accepts a valid final JSON answer when the last terminal event is `turn.completed`, even if the process then lingers until its deadline. Actual timeout, wall time, usage and native-tool events remain recorded. Partial/error turns remain rejected. The default planner, tool planner and recipient comparison share this receiver; this is an infrastructure fix, not a new motion policy.

`result.json` compares parent c5708a0 with the candidate using actual Python child processes. Completed-then-lingering changes from ERROR to ANSWER_RETURNED; partial and completed-then-failed remain ERROR. Fixture token fields are synthetic, explicitly labelled, and not counted as model usage. Three regression tests pass.

Status: **OFFLINE_PROCESS_VERIFIED**. New model calls, physical simulations and hardware commands: **0**. No measured latency or Token saving: the receiver still waits until the original deadline. Live model reuse and task benefit remain untested because the authorized model budget was exhausted. No historical result is relabelled.

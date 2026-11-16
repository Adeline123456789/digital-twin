# Cycle 4: measured planner overhead

Cycle 3's real attempt remains pending the existing supported-disable condition.
This independent cycle addresses the measured 16,802 input tokens for a bounded
right-arm JSON planner call. It spends one of the six system improvement cycles,
not an additional calibration round.

Installed `codex-cli 0.162.1` bundled model metadata contains approximately
18,258 serialized characters of Luna base instructions and 21,669 for Astra.
Those are character counts, not token estimates. Prior removal of the skills
catalog reduced some overhead but left this generic coding role and tool context.

Read the official [configuration schema](https://github.com/openai/codex/blob/main/codex-rs/core/config.schema.json)
and [configuration loader source](https://github.com/openai/codex/blob/main/codex-rs/core/src/config/mod.rs):
`model_instructions_file` is read as a nonempty file and supplies a custom base
instruction source. The official schema discourages overriding it for ordinary
Codex coding work because model performance may degrade. This experiment concerns
only the repository's bounded decision component and tests that risk; it does not
change this research agent or any global configuration.

The sole paired change is the invocation-local base instruction source: a short
JSON decision role instead of general coding instructions. Both groups keep the
same task prompt, robot protocol, read-only sandbox, tool permissions, catalog
setting, bounded action validator, native simulator and complete recorder.
No safety instructions or hardware gates are removed. Any tool use invalidates
the decision before a physics run. Code reuse is the existing `ask`, `run_case`
and Episode verification/publishing path; no framework or dependency is added.

Four calls are preregistered in `design.json`, counterbalancing profile order
between requested Luna and Astra. Each valid decision gets one separately recorded
physical simulation from the same folded entry. Both models see the same historical
negative example and privileged observation, not a successful task answer. This is
a known-task harness cost experiment, not student learning or unseen exploration.
Promotion requires successful valid collection in both candidate pairs and fewer
measured input tokens. All failures, timeouts and unknown usage remain visible.

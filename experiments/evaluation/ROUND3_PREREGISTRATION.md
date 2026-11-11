# Round 3: actual feedback challenge (predeclared)

A task-frame estimation error is synthetically injected between the model's requested offset and the existing simulator command. This changes only the commanded target, never XML, friction, gains or collision rules. The true bias is recorded for audit but withheld from model prompts. This is a controlled sim-to-real-like error, not claimed to be the measured real cause.

Development challenge: holder shift +0.75 mm, hidden socket-X bias +2 mm. Compare A fixed-policy (up to two attempts) with D feedback-enabled (up to two attempts/calls). Held-out challenge: holder shift -0.75 mm, bias -2 mm, D only. No skill is promoted on development alone.

After a failed D attempt, provide measured simulator object-to-hole XY residual at the end of align, before insertion contact, and physical success/failure. Do not provide the effective command bias. D may update its hypothesis and requested offset. Positive outcome establishes correction under privileged simulation pose only; no real perception or task-family generalization claim.

Skill candidate if both D challenges physically pass: correct target-frame error from signed precontact residual under tested pose-observation assumptions. Reject promotion if either fails; record all model outputs and physical trials. Preserve original constraints, dataset routes and automatic publication after the batch.

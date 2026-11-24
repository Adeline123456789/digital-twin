# Full-Task Residual Expansion Plan V1

Expansion is stage-gated; later files are not implemented before their prerequisite passes.

1. Reproduce the local V2 History PPO across five frozen training seeds and 64 new paired conditions.
2. Audit and replay the complete frozen nominal reference from `APPROACH` through `RETREAT`.
3. Test whether post-grasp displacement is identifiable from proprioception history, adding right-wrist RGB only if proprioception fails and RGB passes its own audit.
4. Freeze the controlled XY perturbation range and separately audit whether contact-mediated drift is reproducible without changing friction.
5. Train three fixed-budget full-episode residual policies, freeze final checkpoints and evaluate B0/B1/B2/B3 on paired full episodes.
6. Publish the final `GO`, `NO-GO` or `INCONCLUSIVE` report and stop. Do not begin real transfer automatically.

At every stage, retain the frozen nominal, action limits, left-arm invariance, native success detector, complete Episode validator and zero privileged Student inputs. The detailed thresholds and failure taxonomy are in [MINIMAL_RESEARCH_TASKBOOK_V3.md](MINIMAL_RESEARCH_TASKBOOK_V3.md).

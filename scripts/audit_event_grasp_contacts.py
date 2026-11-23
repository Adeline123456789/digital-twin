"""Replay frozen simulated grasp commands and inspect black-charger contacts at every substep."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['MUJOCO_GL'] = 'egl'
from learn import DATA, ROOT, SIM, digest, save
sys.path.insert(0, str(SIM / 'src'))


def black_robot_pair(names):
    return 'task/black_charger' in names and any(name.startswith('right/') for name in names)


def audit(raw):
    import mujoco
    import numpy as np
    import yaml
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.config import load
    from piper_sim.mixed_recording import configure_rigid_collection
    started = time.monotonic()
    rows = [json.loads(line) for line in (raw / 'trajectory.jsonl').read_text().splitlines()]
    sim = MujocoBackend(dict(load(), task_scene=yaml.safe_load((raw / 'task.yaml').read_text())))
    configure_rigid_collection(sim)
    with np.load(raw / 'initial_state.npz') as initial:
        for key in ('qpos', 'qvel', 'ctrl'):
            getattr(sim.data, key)[:] = initial[key]
    mujoco.mj_forward(sim.model, sim.data)
    geom_body_names = [sim.model.body(sim.model.geom_bodyid[g]).name for g in range(sim.model.ngeom)]
    contact_steps, penetration, state_error, substeps = 0, 0., 0., 0
    first = None
    try:
        for i, row in enumerate(rows):
            sim.apply(sim.layout.command(row['action'], sim.sim_time, i))
            for _ in range(sim.decimation * sim.substeps):
                mujoco.mj_step(sim.model, sim.data)
                mujoco.mj_forward(sim.model, sim.data)
                substeps += 1
                contacts = [c for c in sim.data.contact if c.dist <= 0 and
                            black_robot_pair([geom_body_names[c.geom1], geom_body_names[c.geom2]])]
                if contacts:
                    contact_steps += 1
                    penetration = max(penetration, max(-c.dist for c in contacts))
                    if first is None:
                        first = dict(frame=i, phase=row['phase'], time=sim.sim_time)
            sim.task.metrics(sim.task_adapter.task_poses(sim.data, sim.task), sim.dt * sim.decimation)
            state_error = max(state_error, float(np.max(abs(sim.read().position - row['q']))))
    finally:
        sim.close()
    result = dict(scope='Simulation replay of the recorded prefix; no expert joints or real contact measurements',
                  source_trial=str(raw.relative_to(DATA)), frames=len(rows), integrator_steps=substeps,
                  black_right_arm_contact_steps=contact_steps, max_black_robot_penetration_m=penetration,
                  first_black_robot_contact=first, max_state_error_vs_recording=state_error,
                  checked_time_semantics='Every refreshed integrator endpoint; not a continuous-time mathematical guarantee',
                  physical_steps=substeps, planner_calls=0, hardware_executions=0, wall_s=time.monotonic() - started,
                  evidence=[dict(dataset_relative_path=str(p.relative_to(DATA)), sha256=digest(p))
                            for p in (raw / 'trajectory.jsonl', raw / 'initial_state.npz', raw / 'task.yaml')])
    if state_error > 1e-5:
        result['decision'] = 'UNKNOWN_REPLAY_MISMATCH'
    else:
        result['decision'] = 'NO_BLACK_CONTACT_OBSERVED' if contact_steps == 0 else 'BLACK_CONTACT_OBSERVED'
    save(ROOT / 'experiments/event_residual_mvp/grasp_contact_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    audit(parser.parse_args().trial)

"""Bounded diagnostic replay of a frozen grasp prefix; never a formal Episode or real control."""
import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import yaml

from learn import SIM, digest, save
from trajectory_contract import load_reference

sys.path.insert(0, str(SIM / 'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.mixed_recording import configure_rigid_collection


def in_friction_cone(normal, direction, friction):
    """Analytic point-contact criterion; finite-pad moments are deliberately excluded."""
    cosine = np.dot(normal, direction) / (np.linalg.norm(normal) * np.linalg.norm(direction))
    return bool(cosine >= 1 / np.sqrt(1 + friction**2))


def contact_snapshot(sim):
    model, data = sim.model, sim.data
    yellow = model.body('task/yellow_plug').id
    contacts = []
    for c in data.contact:
        bodies = [int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
        if yellow not in bodies:
            continue
        other = 1 - bodies.index(yellow)
        name = model.body(bodies[other]).name
        if not name.startswith('right/'):
            continue
        normal = np.asarray(c.frame[:3]) * (1 if bodies[0] == yellow else -1)
        contacts.append(dict(robot_body=name, position_m=c.pos.tolist(),
            object_outward_normal=normal.tolist(), distance_m=float(c.dist),
            contact_dim=int(c.dim), sliding_friction=float(c.friction[0]),
            geom_names=[model.geom(g).name for g in (c.geom1, c.geom2)]))
    pairs = []
    for i, first in enumerate(contacts):
        for j, second in enumerate(contacts[i+1:], i+1):
            direction = np.array(second['position_m']) - first['position_m']
            if first['robot_body'] == second['robot_body'] or np.linalg.norm(direction) < 1e-6:
                continue
            passes = (in_friction_cone(-np.array(first['object_outward_normal']), direction,
                                      first['sliding_friction']) and
                      in_friction_cone(-np.array(second['object_outward_normal']), -direction,
                                      second['sliding_friction']))
            pairs.append(dict(indices=[i, j], antipodal_point_contact=passes))
    return dict(contacts=contacts, opposing_body_pairs=pairs,
                any_antipodal_pair=any(p['antipodal_point_contact'] for p in pairs))


def probe(trial, reference, output, contact_dim):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((reference / 'manifest.json').read_text())
    assert digest(trial / 'initial_state.npz') == manifest['source_initial_state_sha256']
    timeline, targets, phases = load_reference(reference)
    lift_indices = [i for i, phase in enumerate(phases) if phase == 'lift']
    stop = timeline[max(lift_indices) + 1]
    sim = MujocoBackend(dict(load(), task_scene=yaml.safe_load((trial / 'task.yaml').read_text())))
    started = time.monotonic()
    result = dict(reference_sha256=manifest['trajectory_id'], result='infrastructure_error',
                  task_success=None, diagnostic_only=True, privileged_information_used=True,
                  real_commands=0, contact_dim=contact_dim, root_cause_status='UNKNOWN')
    history, phase_ends = [], {}
    try:
        result['base_profile'] = configure_rigid_collection(sim)
        yellow = sim.model.body('task/yellow_plug').id
        selected = np.flatnonzero(sim.model.geom_bodyid == yellow)
        sim.model.geom_condim[selected] = contact_dim
        result['changed_geoms'] = len(selected)
        with np.load(trial / 'initial_state.npz') as initial:
            for field in ('qpos', 'qvel', 'ctrl'):
                getattr(sim.data, field)[:] = initial[field]
        mujoco.mj_forward(sim.model, sim.data)
        initial_z = float(sim.data.body('task/yellow_plug').xpos[2])
        assert sim.arms == ('right',)
        index = -1
        next_observation = 1/30
        worst = 0.
        while sim.sim_time < stop:
            current = min(np.searchsorted(timeline, sim.sim_time, side='right') - 1, len(timeline) - 1)
            if current != index:
                index = current
                q = np.r_[np.deg2rad(targets[index, :6]), targets[index, 6] * .095]
                sim.apply(sim.layout.command(q, sim.sim_time, index))
            mujoco.mj_step(sim.model, sim.data)
            mujoco.mj_forward(sim.model, sim.data)
            worst = max(worst, -min((c.dist for c in sim.data.contact), default=0.))
            if worst > .0002:
                result.update(result='failure', reason='Original penetration bound exceeded')
                break
            if sim.sim_time + 1e-9 >= next_observation:
                next_observation += 1/30
                row = dict(time_s=sim.sim_time, phase=phases[index],
                           plug_xyz_m=sim.data.body('task/yellow_plug').xpos.tolist(),
                           measured_gripper_width_m=float(sim.read().position[-1]),
                           geometry=contact_snapshot(sim))
                history.append(row)
                phase_ends[phases[index]] = row
        else:
            result['result'] = 'diagnostic_completed'
        result.update(lift_height_m=float(sim.data.body('task/yellow_plug').xpos[2] - initial_z),
                      simulated_seconds=sim.sim_time, max_penetration_m=worst,
                      phase_ends=phase_ends)
    finally:
        sim.close()
        result['wall_seconds'] = time.monotonic() - started
        (output / 'trace.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in history))
        result['trace_sha256'] = digest(output / 'trace.jsonl')
        save(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contact-dim', type=int, choices=[3, 6], required=True)
    args = parser.parse_args()
    result = probe(args.trial, args.reference, args.output, args.contact_dim)
    print(json.dumps(result, indent=2))

"""Replay the nominal commands from the archived simulation initial state and measure clearance."""
from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import sys

import coal
import mujoco
import numpy as np
import yaml

from learn import SIM
sys.path.insert(0, str(SIM / "src"))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.mixed_recording import configure_rigid_collection


class BodyClearances:
    """Convex outer envelopes give conservative 3-D surface distances in metres."""

    def __init__(self, sim):
        m = sim.model
        bodies = set()
        for side in sim.arms:
            bodies.update(i for i in range(m.nbody) if m.body(i).name.startswith(side + "/"))
        self.plug = m.body("task/yellow_plug").id
        self.charger = m.body("task/black_charger").id
        self.board = m.body("task/board").id
        bodies.update([self.plug, self.charger, self.board])
        self.hulls = {}
        self.finger_vertices = {}
        self.radii = np.zeros(m.nbody)
        corners = np.array(list(product((-1, 1), repeat=3)))
        for body in sorted(bodies):
            pieces = []
            for gid in np.flatnonzero((m.geom_bodyid == body) & ((m.geom_contype | m.geom_conaffinity) != 0)):
                if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[gid]
                    start = m.mesh_vertadr[mid]
                    vertices = m.mesh_vert[start:start + m.mesh_vertnum[mid]]
                else:
                    center, half_size = m.geom_aabb[gid].reshape(2, 3)
                    vertices = center + corners * half_size
                matrix = np.empty(9)
                mujoco.mju_quat2Mat(matrix, m.geom_quat[gid])
                pieces.append(vertices @ matrix.reshape(3, 3).T + m.geom_pos[gid])
            if pieces:
                vertices = np.unique(np.vstack(pieces), axis=0)
                if m.body(body).name in ("right/link7", "right/link8"):
                    self.finger_vertices[body] = vertices
                points = coal.StdVec_Vec3s()
                for vertex in vertices:
                    points.append(vertex)
                self.hulls[body] = coal.Convex.convexHull(points, True, "Qt")
                self.radii[body] = np.max(np.linalg.norm(vertices, axis=1))
        self.robot = np.array(sorted(self.hulls.keys() - {self.board, self.charger, self.plug}))
        self.moving = np.append(self.robot, self.plug)
        self.request = coal.DistanceRequest()
        self.result = coal.DistanceResult()
        self.minima = {}
        self.model = m

    def observe(self, data, phase):
        def pose(i):
            return coal.Transform3s(data.xmat[i].reshape(3, 3), data.xpos[i])

        for target, moving in ((self.charger, self.moving), (self.board, self.robot)):
            name = self.model.body(target).name
            current = self.minima.get(name)
            cutoff = current["lower_bound_m"] if current else np.inf
            lower = np.linalg.norm(data.xpos[moving] - data.xpos[target], axis=1) - self.radii[moving] - self.radii[target]
            for i in moving[lower < cutoff]:
                self.result.clear()
                distance = coal.distance(self.hulls[i], pose(i), self.hulls[target], pose(target), self.request, self.result)
                if distance < cutoff:
                    cutoff = distance
                    self.minima[name] = dict(lower_bound_m=distance, nearest_body=self.model.body(i).name,
                                             phase=phase, time_s=data.time)


def audit(trial: Path) -> dict:
    cfg = yaml.safe_load((trial / "task.yaml").read_text())
    sim = MujocoBackend(dict(load(), task_scene=cfg), cameras=False)
    try:
        configure_rigid_collection(sim)
        with np.load(trial / "initial_state.npz") as state:
            if "integration" in state:
                mujoco.mj_setState(sim.model, sim.data, state["integration"], int(state["spec"]))
            else:
                for key in ("qpos", "qvel", "ctrl"):
                    getattr(sim.data, key)[:] = state[key]
        m, data = sim.model, sim.data
        mujoco.mj_forward(m, data)
        geometry = BodyClearances(sim)
        right = sim.indices["right"]
        left = sim.indices["left"]
        start = sim.read().position
        last = start
        peaks = dict(command_speed_deg_s=0., actual_speed_deg_s=0., insert_j5_command_deg=-180.,
                     insert_j5_actual_deg=-180., penetration_m=0., left_joint_error_rad=0.,
                     left_width_error_m=0., state_replay_error_si=0., plug_replay_error_m=0.)
        endpoint_xy = {}
        shoulder_contacts = {}
        tip_heights = {}
        phase_end_right_deg = {}
        plug_start_z = float(data.xpos[geometry.plug, 2])
        lift_height = 0.0
        phases = []
        preinsert_hold = {}
        nsteps = sim.substeps * sim.decimation
        frame_dt = m.opt.timestep * nsteps
        count = 0
        outlet_local = sim.task.outlets[cfg["target_index"]]
        geometry.observe(data, "initial")
        with (trial / "trajectory.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                action = np.asarray(row["action"])
                phase = row["phase"]
                if not phases or phase != phases[-1]:
                    phases.append(phase)
                if count == 0:
                    first = action.copy()
                np.testing.assert_array_equal(action[:7], [0.] * 6 + [.060])
                peaks["command_speed_deg_s"] = max(peaks["command_speed_deg_s"],
                    np.rad2deg(abs(action[7:13] - last[7:13])).max() / frame_dt)
                if phase == "transport":
                    if not np.array_equal(action, last) or not preinsert_hold:
                        preinsert_hold["first_frame"] = count
                    preinsert_hold["last_frame"] = count
                in_insert = phase in ("transport", "preinsert", "align", "insert")
                if in_insert:
                    # Transport ends at preinsert. Including the whole carry is conservative.
                    peaks["insert_j5_command_deg"] = max(peaks["insert_j5_command_deg"], np.rad2deg(action[11]))
                sim.apply(sim.layout.command(action, data.time, count))
                for _ in range(nsteps):
                    mujoco.mj_step(m, data)
                    mujoco.mj_forward(m, data)
                    if data.ncon:
                        peaks["penetration_m"] = max(peaks["penetration_m"], -float(data.contact.dist.min()))
                    peaks["actual_speed_deg_s"] = max(peaks["actual_speed_deg_s"],
                        np.rad2deg(abs(data.qvel[right["dof"][:6]])).max())
                    if in_insert:
                        peaks["insert_j5_actual_deg"] = max(peaks["insert_j5_actual_deg"], np.rad2deg(data.qpos[right["qpos"][4]]))
                    peaks["left_joint_error_rad"] = max(peaks["left_joint_error_rad"], abs(data.qpos[left["qpos"][:6]]).max())
                    peaks["left_width_error_m"] = max(peaks["left_width_error_m"], abs(data.qpos[left["qpos"][6:]].sum() - .060))
                    geometry.observe(data, phase)
                metrics = sim.task.metrics(sim.task_adapter.task_poses(data, sim.task), frame_dt)
                q = sim.read().position
                plug = data.xpos[geometry.plug]
                peaks["state_replay_error_si"] = max(peaks["state_replay_error_si"], abs(q - row["q"]).max())
                peaks["plug_replay_error_m"] = max(peaks["plug_replay_error_m"], abs(plug - row["plug"]).max())
                phase_end_right_deg[phase] = np.rad2deg(action[7:13]).tolist()
                if phase == "lift":
                    lift_height = max(lift_height, float(plug[2] - plug_start_z))
                if phase == "grasp":
                    # Retain the final grasp frame in plug-local coordinates. The
                    # shoulder geometry and the encoder give independent evidence.
                    shoulder_contacts = {}
                    plug_rotation = data.xmat[geometry.plug].reshape(3, 3)
                    for finger, vertices in geometry.finger_vertices.items():
                        world = vertices @ data.xmat[finger].reshape(3, 3).T + data.xpos[finger]
                        tip_heights[m.body(finger).name] = float(((world - plug) @ plug_rotation[:, 2]).min())
                    for contact in data.contact:
                        if contact.efc_address < 0:
                            continue
                        ids = m.geom_bodyid[[contact.geom1, contact.geom2]]
                        if geometry.plug not in ids:
                            continue
                        other = m.body(ids[1] if ids[0] == geometry.plug else ids[0]).name
                        if other in ("right/link7", "right/link8"):
                            point = data.xmat[geometry.plug].reshape(3, 3).T @ (contact.pos - plug)
                            plug_gid = contact.geom1 if ids[0] == geometry.plug else contact.geom2
                            shoulder_contacts.setdefault(other, []).append(dict(
                                plug_geom=m.geom(plug_gid).name, point_m=point.tolist()))
                if phase in ("transport", "align", "insert"):
                    strip = data.body("task/strip")
                    target = strip.xpos + strip.xmat.reshape(3, 3)[:, :2] @ outlet_local
                    endpoint_xy[phase] = (plug[:2] - target[:2]).tolist()
                last = action
                count += 1
        return dict(frames=count, substeps=count*nsteps, phases=phases, peaks=peaks, clearance=geometry.minima,
                    native_success=metrics["success"], lift_displacement_m=lift_height,
                    final_grasp_contacts_plug_local=shoulder_contacts, fingertip_min_plug_local_z_m=tip_heights,
                    phase_end_right_deg=phase_end_right_deg, preinsert_hold_zero_based=preinsert_hold,
                    initial_state_si=start.tolist(), first_action_si=first.tolist(),
                    last_action_si=last.tolist(), final_state_si=q.tolist(), plug_minus_outlet_world_xy_m=endpoint_xy)
    finally:
        sim.close()


def verify_reference(replay, recorded_success):
    peaks = replay["peaks"]
    home = [0.] * 6 + [.060] + [0.] * 6 + [.040]
    for key in ("initial_state_si", "first_action_si"):
        np.testing.assert_array_equal(replay[key], home)
    np.testing.assert_array_equal(replay["last_action_si"][7:13], np.zeros(6))
    np.testing.assert_allclose(replay["final_state_si"][7:13], np.zeros(6), rtol=0., atol=1.e-6)
    if replay["phases"] != ["prepare_gripper", "approach", "descend", "grasp", "lift",
                            "transport", "align", "insert", "release", "retreat", "fold"]:
        raise ValueError("Incomplete nominal phase sequence")
    if not (recorded_success and replay["native_success"]):
        raise ValueError("Native task success required on recording and replay")
    for key, bound in dict(command_speed_deg_s=12., actual_speed_deg_s=12.,
                           insert_j5_command_deg=68., insert_j5_actual_deg=68.,
                           penetration_m=.0002, state_replay_error_si=1.e-6, plug_replay_error_m=1.e-6,
                           left_joint_error_rad=1.e-8, left_width_error_m=1.e-8).items():
        if not peaks[key] <= bound:
            raise ValueError(f"{key} = {peaks[key]} exceeds {bound}")
    clearance = replay["clearance"]
    if clearance["task/black_charger"]["lower_bound_m"] < .008 or clearance["task/board"]["lower_bound_m"] <= 0.:
        raise ValueError(f"Insufficient collision-geometry clearance: {clearance}")

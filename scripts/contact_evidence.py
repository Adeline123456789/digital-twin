"""Inspect an archived simulation frame; never advances physics or edits the source model."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from episode_evidence import locate, frame_at


def integration_snapshot(model, data):
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return dict(mujoco_version=mujoco.__version__, spec=spec, values=state.tolist())


def model_dependencies(xml):
    tree = ET.parse(xml).getroot()
    compiler = tree.find('compiler')
    directory = Path(compiler.get('meshdir', '') if compiler is not None else '')
    deps = dict(model=xml)
    # Only referenced meshes, not the whole assets tree or any videos.
    for mesh in tree.findall('./asset/mesh'):
        if mesh.get('file'):
            path = xml.parent / directory / mesh.get('file')
            deps['mesh:' + mesh.get('name', mesh.get('file'))] = path
    return deps


def restore(model, data, row, extra, initial):
    snapshot = extra.get('physics_state')
    if snapshot:
        if snapshot['mujoco_version'] != mujoco.__version__:
            raise ValueError('Recorded integration state requires its original MuJoCo version')
        mujoco.mj_setState(model, data, np.asarray(snapshot['values']), snapshot['spec'])
        scope = 'RECORDED_INTEGRATION_STATE'
    else:
        with np.load(initial) as state:
            for key in ('qpos', 'qvel', 'ctrl'):
                getattr(data, key)[:] = state[key]
        arms = [s for s in ('left', 'right') if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, s+'/joint1') >= 0]
        if len(row['q']) != 7 * len(arms):
            raise ValueError('Recorded robot layout does not match archived model')
        for i, side in enumerate(arms):
            q = row['q'][7*i:7*i+7]
            for j, value in enumerate([*q[:6], q[6]/2, q[6]/2], 1):
                data.qpos[model.joint(f'{side}/joint{j}').qposadr[0]] = value
            for name, value in zip([*[f'joint{j}' for j in range(1,7)], 'gripper'], row['action'][7*i:7*i+7]):
                data.ctrl[model.actuator(side+'/'+name).id] = value
        for name, pose in extra['objects'].items():
            body = model.body('task/'+name).id
            if model.body_jntnum[body]:
                joint = model.body_jntadr[body]
                if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                    raise ValueError('Only recorded free-object poses can be reconstructed')
                adr = model.jnt_qposadr[joint]
                data.qpos[adr:adr+7] = [*pose['position'], *pose['quaternion_wxyz']]
        data.qvel[:] = extra['qvel']
        data.time = row['time']
        scope = 'RECONSTRUCTED_PARTIAL_STATE'
    mujoco.mj_forward(model, data)
    return scope


def visible_mesh_distance(model, data, body, position):
    import trimesh
    opts = mujoco.MjvOption()
    nearest = []
    for geom in np.flatnonzero(model.geom_bodyid == body):
        if (not opts.geomgroup[model.geom_group[geom]] or model.geom_rgba[geom,3] == 0
                or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = model.geom_dataid[geom]
        vs, fs = model.mesh_vertadr[mid], model.mesh_faceadr[mid]
        mesh = trimesh.Trimesh(vertices=model.mesh_vert[vs:vs+model.mesh_vertnum[mid]],
                              faces=model.mesh_face[fs:fs+model.mesh_facenum[mid]], process=False)
        point = (position-data.geom_xpos[geom]) @ data.geom_xmat[geom].reshape(3,3)
        _, distance, _ = trimesh.proximity.closest_point_naive(mesh, point[None,:])
        nearest.append(dict(geom=model.geom(geom).name, distance_mm=float(distance[0])*1000))
    return sorted(nearest, key=lambda x:x['distance_mm'])[:2]


def inspect_state(model, data, focus_body='task/yellow_plug', surfaces=True):
    focus = model.body(focus_body).id
    contacts = []
    for i, contact in enumerate(data.contact):
        geoms = [contact.geom1, contact.geom2]
        if focus not in model.geom_bodyid[geoms]: continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        item = dict(pair=[model.geom(g).name for g in geoms], distance_m=float(contact.dist),
            effective_includemargin_m=float(contact.includemargin),
            solver_active=bool(contact.efc_address >= 0), constraint_address=int(contact.efc_address),
            exclude=int(contact.exclude), condim=int(contact.dim), friction=contact.friction.tolist(),
            solved_force_contact_frame_N_Nm=force.tolist(), geoms=[])
        for geom in geoms:
            g = dict(name=model.geom(geom).name, type=int(model.geom_type[geom]),
                margin_m=float(model.geom_margin[geom]), gap_m=float(model.geom_gap[geom]),
                friction=model.geom_friction[geom].tolist(), group=int(model.geom_group[geom]),
                rgba=model.geom_rgba[geom].tolist(), contype=int(model.geom_contype[geom]),
                conaffinity=int(model.geom_conaffinity[geom]),
                mesh_collision='convex hull' if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH else None)
            if surfaces:
                g['nearest_visible_mesh_surface'] = visible_mesh_distance(model, data, model.geom_bodyid[geom], contact.pos)
            item['geoms'].append(g)
        contacts.append(item)
    actuators = []
    for i in range(model.nu):
        actuators.append(dict(name=model.actuator(i).name, transmission_type=int(model.actuator_trntype[i]),
            ctrl=float(data.ctrl[i]), force=float(data.actuator_force[i]),
            force_limited=bool(model.actuator_forcelimited[i]), forcerange=model.actuator_forcerange[i].tolist(),
            ctrl_limited=bool(model.actuator_ctrllimited[i]), ctrlrange=model.actuator_ctrlrange[i].tolist(),
            gear=model.actuator_gear[i].tolist()))
    return dict(contacts=contacts, actuators=actuators,
        equality_constraints=[dict(name=model.equality(i).name, type=int(model.eq_type[i]),
            active=bool(data.eq_active[i]), object_ids=[int(model.eq_obj1id[i]),int(model.eq_obj2id[i])]) for i in range(model.neq)],
        force_state=dict(xfrc_applied_nonzero=bool(np.any(data.xfrc_applied)),
                         qfrc_applied_nonzero=bool(np.any(data.qfrc_applied))),
        contact_override=dict(enabled=bool(model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)),
            margin_m=float(model.opt.o_margin), friction=model.opt.o_friction.tolist()),
        mesh_distance_limit='Contact-point distance to visible mesh; not silhouette gap or full surface-to-surface distance',
        units_limit='Prismatic actuator N is not real gripper motor N.m; no identified conversion')


def inspect_frame(ident, frame, cache, surfaces=True):
    from importlib.metadata import version
    report, episode, raw = locate(ident)
    stats = episode/'proprio_stats'
    deps = dict(report=report, trajectory=raw/'trajectory.jsonl', privileged=stats/'privileged.jsonl',
                initial=stats/'initial_state.npz', analyzer=Path(__file__),
                reader=Path(__file__).with_name('episode_evidence.py'), **model_dependencies(stats/'model.xml'))
    def compute():
        row = frame_at(deps['trajectory'], frame)
        extra = frame_at(deps['privileged'], frame)
        if extra.get('phase') != row['phase'] or extra['frame'] != frame:
            raise ValueError('Frame/phase mismatch')
        model = mujoco.MjModel.from_xml_path(str(stats/'model.xml'))
        data = mujoco.MjData(model)
        scope = restore(model, data, row, extra, deps['initial'])
        result = inspect_state(model, data, surfaces=surfaces)
        result.update(episode_id=ident, frame=frame, phase=row['phase'], time_s=row['time'],
            state_source=scope, force_source='mj_forward solver on restored state, not measured real force',
            recorded_contact_count=len(extra['contacts']),
            historical_external_forces_status='RECORDED' if scope == 'RECORDED_INTEGRATION_STATE' else 'UNKNOWN',
            historical_direct_state_edits='UNKNOWN; a snapshot cannot exclude earlier object teleports or callbacks',
            geometry_provenance='Archived XML and current referenced mesh hashes; original external mesh identity was not recorded',
            reconstruction_limit='Historical width-only samples assume equal finger displacement; full integration state avoids this assumption',
            root_cause='UNKNOWN', privileged_information_used=True, simulation_steps=0)
        return result
    return cache.get('contact_frame_v1', deps, dict(frame=frame, surfaces=surfaces,
        mujoco_version=mujoco.__version__, trimesh_version=version('trimesh')), compute)

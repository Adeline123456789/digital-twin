"""Synthetic RGB sequences; privileged projection/depth are labels only."""
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent.residual_env import DATA, ROOT, SIM
from learn import digest


def project(model, data, camera, points):
    cid = model.camera(camera).id
    optical = (points - data.cam_xpos[cid]) @ data.cam_xmat[cid].reshape(3, 3) * [1, -1, -1]
    f = 240 / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
    return optical[:, :2] / optical[:, 2:3] * f + [320, 240], optical[:, 2]


def build():
    design = json.loads((ROOT / 'experiments/event_residual_mvp_v2/perception_design.json').read_text())
    output = DATA / 'self_evolution/event_residual_mvp_v2/perception'
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    for source_index, episode in enumerate(design['sources']):
        raw = DATA / 'self_evolution' / episode
        stats = DATA / 'complete_sim/succeed' / episode / 'proprio_stats'
        task = yaml.safe_load((raw / 'task.yaml').read_text())
        rows = [json.loads(line) for line in (raw / 'trajectory.jsonl').read_text().splitlines()]
        frames = sorted({0, *range(120, 441, 20), *[i for i, row in enumerate(rows) if row['phase'] in ('align', 'insert')][::8]})
        with (stats / 'privileged.jsonl').open() as stream:
            states = {i: json.loads(line)['physics_state'] for i, line in enumerate(stream) if i in frames}
        model = mujoco.MjModel.from_xml_path(str(stats / 'model.xml'))
        data = mujoco.MjData(model)
        cam_pos, cam_quat, colors = model.cam_pos.copy(), model.cam_quat.copy(), model.geom_rgba.copy()
        geom = model.geom('task/yellow_plug/pin_earth').id
        mesh = model.geom_dataid[geom]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        # Earth/top pin tip and its matching socket opening in the native task frame.
        plug_config = yaml.safe_load((SIM / 'configs/tasks/connector.yaml').read_text())['plugs']['yellow_plug']
        plug_config.update(task['yellow_plug'])
        hole_local = np.r_[np.asarray(task['first_module_xy_m']) + [3 * task['outlet_pitch_m'], plug_config['earth_offset_m']],
                           task['strip_size_m'][2] - task['key_projection_m']]
        with mujoco.Renderer(model, height=480, width=640) as renderer, mujoco.Renderer(model, height=480, width=640) as depth_renderer:
            depth_renderer.enable_depth_rendering()
            for seed in design['sequence_seeds'][source_index]:
                rng = np.random.default_rng(seed)
                model.cam_pos[:] = cam_pos + rng.uniform(-.002, .002, cam_pos.shape)
                for cid in range(model.ncam):
                    model.cam_quat[cid] = (Rotation.from_quat(cam_quat[cid], scalar_first=True) *
                        Rotation.from_euler('xyz', rng.uniform(-1., 1., 3), degrees=True)).as_quat(scalar_first=True)
                model.geom_rgba[:, :3] = np.clip(colors[:, :3] * rng.uniform(.85, 1.15, (model.ngeom, 1)), 0, 1)
                images, labels, visibility, meta = [], [], [], []
                for frame in frames:
                    snapshot = states[frame]
                    mujoco.mj_setState(model, data, np.asarray(snapshot['values']), snapshot['spec'])
                    mujoco.mj_forward(model, data)
                    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                    plug = data.body('task/yellow_plug')
                    local_z = ((world - plug.xpos) @ plug.xmat.reshape(3, 3))[:, 2]
                    pin = world[np.isclose(local_z, local_z.min(), atol=1e-7)].mean(axis=0)
                    strip = data.body('task/strip')
                    hole = hole_local @ strip.xmat.reshape(3, 3).T + strip.xpos
                    for camera in ('overhead', 'right_wrist'):
                        renderer.update_scene(data, camera=camera)
                        rgb = renderer.render()
                        depth_renderer.update_scene(data, camera=camera)
                        depth = depth_renderer.render()
                        uv, z = project(model, data, camera, np.array([pin, hole]))
                        seen = []
                        for key, ((u, v), distance) in enumerate(zip(uv, z)):
                            x, y = int(round(u)), int(round(v))
                            valid = distance > 0 and 1 <= x < 639 and 1 <= y < 479
                            # Raster tolerance: pin surface 1 mm, recessed hole 6 mm.
                            seen.append(bool(valid and abs(distance - depth[y-1:y+2, x-1:x+2].max()) < (.001 if key == 0 else .006)))
                        # Brightness, contrast, image texture, blur and explicit occlusion.
                        image = (rgb.astype(float) - 127) * rng.uniform(.8, 1.2) + 127
                        image = image * rng.uniform(.8, 1.2) + rng.normal(0, 2, image.shape)
                        image = np.clip(image, 0, 255).astype(np.uint8)
                        if rng.random() < .3:
                            image = cv2.GaussianBlur(image, (3, 3), 0)
                        occluded = False
                        if rng.random() < .2:
                            x, y = rng.integers([0, 0], [570, 420]); w, h = rng.integers(20, 70, 2)
                            image[y:y+h, x:x+w] = rng.integers(0, 255, 3)
                            for k, (u, v) in enumerate(uv):
                                if x <= u < x + w and y <= v < y + h:
                                    seen[k], occluded = False, True
                        images.append(cv2.resize(image, (320, 240)))
                        labels.append(uv / 2)
                        visibility.append(seen)
                        meta.append(dict(frame=frame, phase=rows[frame]['phase'], time=rows[frame]['time'], camera=camera, artificial_occlusion=occluded))
                path = output / f'sequence_{seed}.npz'
                np.savez_compressed(path, rgb=np.asarray(images), keypoints=np.asarray(labels), visible=np.asarray(visibility))
                (output / f'sequence_{seed}.json').write_text(json.dumps(meta) + '\n')
                item = dict(seed=seed, source_episode=episode, samples=len(images), split=design['splits'][str(seed)],
                    visible_counts=np.sum(visibility, axis=0).tolist(), dataset_relative_path=str(path.relative_to(DATA)), sha256=digest(path))
                summary.append(item)
                print('PERCEPTION_SEQUENCE', json.dumps(item), flush=True)
    result = dict(sequences=summary, scope='Synthetic rendered sequences from two archived physical simulation episodes; frames are not independent trials',
        labels='Evaluator/training only: native earth pin end and matching outlet-3 hole, depth visibility',
        left_wrist='excluded from Student and training', physics_steps=0)
    (ROOT / 'experiments/event_residual_mvp_v2/perception_dataset.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    build()

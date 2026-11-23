"""Rendered object-pose augmentation for keypoint training, never a physics trial."""
import json
from pathlib import Path
import sys
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent.residual_env import LocalResidualEnv, DATA, ROOT
from build_perception_v2 import project
from learn import digest


def main():
    env = LocalResidualEnv(training=False)
    env.reset(seed=77)
    model, data = env.sim.model, env.sim.data
    geom = model.geom('task/yellow_plug/pin_earth').id
    mesh = model.geom_dataid[geom]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    task = env.sim.task
    earth = next(p for p in task.fixture.parts['yellow_plug'] if p['name'] == 'pin_earth')
    xy = np.asarray(earth['points'])[:, :2].mean(axis=0)
    strip = data.body('task/strip')
    hole = np.r_[task.outlets[3] + xy, task.socket_face_z] @ strip.xmat.reshape(3, 3).T + strip.xpos
    rng = np.random.default_rng(777)
    images, labels, visibility = [], [], []
    with mujoco.Renderer(model, height=480, width=640) as renderer, mujoco.Renderer(model, height=480, width=640) as depths:
        depths.enable_depth_rendering()
        for index in range(256):
            data.qpos[env.plug_qadr:env.plug_qadr + 3] = hole + rng.uniform([-.07, -.07, .03], [.07, .07, .14])
            data.qpos[env.plug_qadr + 3:env.plug_qadr + 7] = Rotation.random(random_state=rng).as_quat(scalar_first=True)
            mujoco.mj_forward(model, data)
            world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            plug = data.body('task/yellow_plug')
            z = ((world - plug.xpos) @ plug.xmat.reshape(3, 3))[:, 2]
            pin = world[np.isclose(z, z.min(), atol=1e-7)].mean(axis=0)
            for camera in ('overhead', 'right_wrist'):
                renderer.update_scene(data, camera=camera); rgb = renderer.render()
                depths.update_scene(data, camera=camera); depth = depths.render()
                uv, distances = project(model, data, camera, np.array([pin, hole]))
                seen = []
                for k, ((u, v), distance) in enumerate(zip(uv, distances)):
                    x, y = int(round(u)), int(round(v))
                    valid = distance > 0 and 1 <= x < 639 and 1 <= y < 479
                    sample = depth[y-1:y+2, x-1:x+2].max() if valid else 0.
                    seen.append(bool(valid and (abs(distance - sample) < .001 if k == 0 else sample >= distance - .002)))
                rgb = np.clip((rgb.astype(float) - 127) * rng.uniform(.8, 1.2) + 127 + rng.normal(0, 3, rgb.shape), 0, 255).astype(np.uint8)
                images.append(cv2.resize(rgb, (320, 240))); labels.append(uv / 2); visibility.append(seen)
    env.close()
    path = DATA / 'self_evolution/event_residual_mvp_v2/perception/auxiliary_training_views.npz'
    np.savez_compressed(path, rgb=np.asarray(images), keypoints=np.asarray(labels), visible=np.asarray(visibility))
    result = dict(dataset_relative_path=str(path.relative_to(DATA)), sha256=digest(path), samples=len(images),
        visible_counts=np.sum(visibility, axis=0).tolist(), seed=777,
        scope='TRAINING ONLY. Static object-pose render augmentation, not collision-validated physics or held-out task performance.',
        reason='Task views have no directly visible earth-pin end; add exposed orientations to train appearance, then independently evaluate actual task occlusion.',
        physics_steps=0, source='unchanged frozen nominal local model; ephemeral data.qpos poses only')
    (ROOT / 'experiments/event_residual_mvp_v2/perception_auxiliary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

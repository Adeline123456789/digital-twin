"""Local frozen-reference MuJoCo task. Privileged state stays in reset/reward/evaluation."""
import json
from pathlib import Path
import sys

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .nominal_reference import ReferencePlayer, content_hash
from .proprio_history import ProprioHistory, FEATURES_PER_FRAME
from .residual_controller import SmoothResidual, inject_xy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from learn import DATA, SIM, digest
sys.path.insert(0, str(SIM / 'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.mixed_recording import configure_rigid_collection, checked_physics_step


class LocalResidualEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, history_frames=15, reward_kind='structured', training=True,
                 visual_features=False, visual_dropout=.5):
        manifest = json.loads((ROOT / 'experiments/event_residual_mvp/nominal_manifest.json').read_text())
        reference = DATA / manifest['dataset_relative_path']
        if digest(reference) != manifest['sha256']:
            raise ValueError('Frozen nominal changed')
        self.player = ReferencePlayer(json.loads(reference.read_text()), manifest['reference_hash'])
        self.raw = DATA / 'self_evolution' / manifest['reference_episode_id']
        stats = DATA / 'complete_sim/succeed' / manifest['reference_episode_id'] / 'proprio_stats'
        task = yaml.safe_load((self.raw / 'task.yaml').read_text())
        self.sim = MujocoBackend(dict(load(), task_scene=task), cameras=False)
        configure_rigid_collection(self.sim)
        self.start = max(i for i, f in enumerate(self.player.frames) if f.phase == 'transport')
        self.end = max(i for i, f in enumerate(self.player.frames) if f.phase == 'retreat')
        with (stats / 'privileged.jsonl').open() as stream:
            for i, line in enumerate(stream):
                if i == self.start:
                    self.initial = json.loads(line)['physics_state']
                    break
        if self.initial['mujoco_version'] != mujoco.__version__:
            raise ValueError('Use the original frozen-reference MuJoCo version')
        self.history_frames = history_frames
        self.reward_kind, self.training = reward_kind, training
        self.visual_features, self.visual_dropout = visual_features, visual_dropout
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                               shape=(FEATURES_PER_FRAME * history_frames + 4 * visual_features,), dtype=np.float32)
        self.strip_id = self.sim.model.body('task/strip').id
        self.strip_position = self.sim.model.body_pos[self.strip_id].copy()
        self.plug_qadr = self.sim.model.joint('task/yellow_plug/free').qposadr[0]
        self.joint_maps = self._robot_joint_maps()
        self.physics_steps = 0
        self.rgb_renderer = None

    def _robot_joint_maps(self):
        # Robot kinematics use frozen joint targets, never a live object/goal pose.
        model = self.sim.model
        data = mujoco.MjData(model)
        ids = self.sim.indices['right']['qpos'][:6]
        dofs = self.sim.indices['right']['dof'][:6]
        body = model.body('right/gripper_center').id
        result = {}
        for i in range(self.start + 1, self.end + 1):
            data.qpos[:] = self.sim.data.qpos
            data.qpos[ids] = self.player.frames[i].nominal_action[7:13]
            mujoco.mj_forward(model, data)
            linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, linear, angular, body)
            jacobian = np.r_[linear[:, dofs], angular[:, dofs]]
            desired_xy = np.zeros((6, 2)); desired_xy[:2] = np.eye(2)
            result[i] = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8 * np.eye(6), desired_xy)
        return result

    def _feedback(self):
        ids = self.sim.indices['right']
        q = self.sim.data.qpos[ids['qpos']]
        v = self.sim.data.qvel[ids['dof']]
        return np.r_[q[:6], q[6] + q[7]], np.r_[v[:6], v[6] + v[7]]

    def _observe(self):
        q, velocity = self._feedback()
        frame = self.player.frames[self.index]
        return self.history.append(position=q, velocity=velocity, nominal_action=frame.nominal_action[7:14],
                                   controller_target=self.last_action[7:14], phase=frame.phase)

    def _metrics(self):
        return self.sim.task.metrics(self.sim.task_adapter.task_poses(self.sim.data, self.sim.task))

    def _with_vision(self, observation):
        if not self.visual_features:
            return observation
        # Optional RGB branch; local inference sees only rendered RGB, never poses.
        if self.step_count % 3 == 0:
            if self.rgb_renderer is None:
                self.rgb_renderer = mujoco.Renderer(self.sim.model, height=480, width=640)
            self.rgb_renderer.update_scene(self.sim.data, camera='overhead')
            features = self.visual_tracker.update(self.rgb_renderer.render(), self.sim.sim_time,
                                                 self.player.frames[self.index].phase)
            self.visual = np.zeros(4, np.float32)
            if features.relative_image_residual is not None:
                self.visual[:] = [*(np.asarray(features.relative_image_residual) / 640), features.feature_score, 1.]
                self.visual_available += 1
            self.visual_samples += 1
        visible = self.visual if self.np_random.random() >= self.visual_dropout else np.zeros(4, np.float32)
        return np.r_[observation, visible].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.model.body_pos[self.strip_id] = self.strip_position
        self.sim.reset()
        if self.visual_features:
            from .observation_adapter import VisualTracker
            self.visual_tracker = VisualTracker()
            if self.rgb_renderer is None:
                self.rgb_renderer = mujoco.Renderer(self.sim.model, height=480, width=640)
            self.rgb_renderer.update_scene(self.sim.data, camera='overhead')
            self.visual_tracker.update(self.rgb_renderer.render(), 0., 'approach')
            # The local task jumps to the archived preinsert state; retain identity,
            # but do not optical-flow across the skipped transport interval.
            self.visual_tracker.previous_gray = self.visual_tracker.previous_time = None
        mujoco.mj_setState(self.sim.model, self.sim.data, np.asarray(self.initial['values']), self.initial['spec'])
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.perturbation = options.get('perturbation') if options else None
        if self.perturbation is None:
            self.perturbation = (dict(socket_xy_mm=self.np_random.uniform(-1., 1., 2).tolist(),
                hand_xy_mm=self.np_random.uniform(-.1, .1, 2).tolist(), yaw_deg=float(self.np_random.uniform(-.5, .5)))
                if self.training else dict(socket_xy_mm=[0., 0.], hand_xy_mm=[0., 0.], yaw_deg=0.))
        p = self.perturbation
        self.sim.model.body_pos[self.strip_id, :2] += np.asarray(p['socket_xy_mm']) / 1000
        grip_rotation = self.sim.data.body('right/gripper_center').xmat.reshape(3, 3)
        self.sim.data.qpos[self.plug_qadr:self.plug_qadr + 3] += grip_rotation @ np.r_[np.asarray(p['hand_xy_mm']) / 1000, 0.]
        quat = self.sim.data.qpos[self.plug_qadr + 3:self.plug_qadr + 7]
        if p['yaw_deg']:
            hand_frame = Rotation.from_matrix(grip_rotation)
            rotated = (hand_frame * Rotation.from_euler('z', p['yaw_deg'], degrees=True)
                       * hand_frame.inv() * Rotation.from_quat(quat, scalar_first=True))
            quat[:] = rotated.as_quat(scalar_first=True)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.index = self.start
        self.history = ProprioHistory(self.history_frames)
        self.smoother = SmoothResidual()
        self.last_action = np.asarray(self.player.frames[self.index].nominal_action).copy()
        self.previous_metrics = self._metrics()
        self.previous_target = np.zeros(2)
        self.correction_energy = self.variation = self.max_penetration = 0.
        self.step_count = 0
        self.visual_available = self.visual_samples = 0
        self.visual = np.zeros(4, np.float32)
        return self._with_vision(self._observe()), {}

    def step(self, action):
        target = np.asarray(action, dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Policy output must be finite XY')
        target = np.clip(target, -1., 1.)
        collision = False
        delta_penalty = 0.
        for _ in range(2):
            if self.index == self.end:
                break
            self.index += 1
            frame = self.player.frames[self.index]
            # Preserve the last insertion correction through finger release; fade during retreat.
            if frame.phase == 'release':
                target = self.previous_target
            elif frame.phase == 'retreat':
                target = np.zeros(2)
            xy = self.smoother.update(target)
            self.last_action = inject_xy(frame.nominal_action, xy, self.joint_maps[self.index],
                                         self.sim.layout.lower, self.sim.layout.upper)
            self.sim.apply(self.sim.layout.command(self.last_action, self.sim.sim_time, self.index))
            check = checked_physics_step(self.sim, .0002)
            self.physics_steps += 1
            self.max_penetration = max(self.max_penetration, check['max_penetration_m'])
            self.correction_energy += float(np.linalg.norm(xy)) / 30
            delta_penalty += float(np.sum((target - self.previous_target) ** 2))
            self.variation += float(np.linalg.norm(self.smoother.velocity)) / 30
            self.previous_target = target.copy()
            observation = self._observe()
            if not check['valid']:
                collision = True
                break
        self.step_count += 1
        current = self._metrics()
        finished = self.index == self.end
        success = bool(finished and not collision and current['success'])
        reward = (50. if success else 0.) - (20. if collision else 0.) - .01
        reward -= .03 * float(np.sum((self.smoother.position / self.smoother.bound) ** 2)) + .01 * delta_penalty
        if self.reward_kind == 'structured':
            reward += 500 * (self.previous_metrics['face_gap_m'] - current['face_gap_m'])
            reward += 2000 * (self.previous_metrics['lateral_error_m'] - current['lateral_error_m'])
            reward -= .1 * check['max_penetration_m'] / .0002
        self.previous_metrics = current
        info = {}
        if finished or collision:
            info = dict(is_success=success, collision=collision, max_penetration_m=self.max_penetration,
                        correction_integral_m_s=self.correction_energy, correction_variation_m=self.variation,
                        duration_s=self.sim.sim_time - self.player.frames[self.start].timestep,
                        perturbation=self.perturbation, final_metrics=current,
                        privileged_student_inputs=0, nominal_reference_hash=self.player.reference_hash)
            info.update(visual_available_samples=self.visual_available, visual_samples=self.visual_samples)
        return self._with_vision(observation), float(reward), bool(finished or collision), False, info

    def close(self):
        if self.rgb_renderer is not None:
            self.rgb_renderer.close()
        self.sim.close()

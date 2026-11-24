"""Full-task P0 residual environment for CONTACT_MEDIATED_DRIFT_V1."""
import json
from pathlib import Path
import sys

import gymnasium as gym
import mujoco
import numpy as np
import yaml

from .full_task_observation import FULL_TASK_FEATURES_PER_FRAME, FullTaskProprioHistory
from .nominal_reference import ReferencePlayer
from .residual_controller import SmoothResidual, inject_xy


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug')
DESIGN = ROOT / 'experiments/residual_fulltask_mvp/contact_drift_design.json'
sys.path.insert(0, str(ROOT / 'scripts'))
from learn import SIM, digest

sys.path.insert(0, str(SIM / 'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.mixed_recording import configure_rigid_collection


def sample_contact_drift(distribution, rng, condition_id):
    intervals = np.asarray(distribution['angle_rad']['intervals'], dtype=float)
    weights = np.asarray(distribution['angle_rad']['weights'], dtype=float)
    interval = intervals[rng.choice(len(intervals), p=weights / weights.sum())]
    return {
        'condition_id': condition_id,
        'phase': distribution['phase'],
        'force_n': float(distribution['force_n']),
        'duration_ms': float(distribution['duration_ms']),
        'angle_rad': float(rng.uniform(*interval)),
    }


class FullTaskContactDriftEnv(gym.Env):
    """Post-grasp policy with deployable P0 history and privileged reward only."""

    metadata = {'render_modes': []}

    def __init__(self, design_path=DESIGN, training=True):
        self.design_path = Path(design_path)
        self.design = json.loads(self.design_path.read_text())
        self.design_sha256 = digest(self.design_path)
        manifest = json.loads((ROOT / 'experiments/event_residual_mvp/nominal_manifest.json').read_text())
        reference_path = DATA / manifest['dataset_relative_path']
        if digest(reference_path) != manifest['sha256']:
            raise ValueError('Frozen nominal changed')
        self.player = ReferencePlayer(json.loads(reference_path.read_text()), manifest['reference_hash'])
        source_id = manifest['reference_episode_id']
        raw = DATA / 'self_evolution' / source_id
        stats = DATA / 'complete_sim/succeed' / source_id / 'proprio_stats'
        task = yaml.safe_load((raw / 'task.yaml').read_text())
        self.sim = MujocoBackend(dict(load(), task_scene=task), cameras=False)
        configure_rigid_collection(self.sim)
        self.start = max(i for i, frame in enumerate(self.player.frames) if frame.phase == 'grasp')
        self.retreat_end = max(i for i, frame in enumerate(self.player.frames) if frame.phase == 'retreat')
        self.episode_end = len(self.player.frames) - 1
        with (stats / 'privileged.jsonl').open() as stream:
            self.initial = next(json.loads(line)['physics_state']
                                for i, line in enumerate(stream) if i == self.start)
        self.anchor = int(self.design['distribution']['anchor_frame'])
        self.training = training
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(FULL_TASK_FEATURES_PER_FRAME * 15,), dtype=np.float32)
        self.right = self.sim.indices['right']
        self.plug_body = self.sim.model.body('task/yellow_plug').id
        self.joint_maps = self._robot_joint_maps()
        self.episode_number = 0

    def _robot_joint_maps(self):
        model = self.sim.model
        data = mujoco.MjData(model)
        ids = self.right['qpos'][:6]
        dofs = self.right['dof'][:6]
        body = model.body('right/gripper_center').id
        maps = {}
        for index in range(self.start + 1, self.retreat_end + 1):
            data.qpos[:] = self.sim.data.qpos
            data.qpos[ids] = self.player.frames[index].nominal_action[7:13]
            mujoco.mj_forward(model, data)
            linear = np.zeros((3, model.nv))
            angular = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, linear, angular, body)
            jacobian = np.r_[linear[:, dofs], angular[:, dofs]]
            desired = np.zeros((6, 2))
            desired[:2] = np.eye(2)
            maps[index] = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-8 * np.eye(6), desired)
        return maps

    def _feedback(self):
        qpos = self.sim.data.qpos[self.right['qpos']]
        qvel = self.sim.data.qvel[self.right['dof']]
        return qpos, qvel

    def _observe(self, frame_index):
        qpos, qvel = self._feedback()
        frame = self.player.frames[frame_index]
        self.observation = self.history.append(
            joint_position=qpos[:6],
            joint_velocity=qvel[:6],
            nominal_action=frame.nominal_action[7:14],
            controller_target=self.last_action[7:14],
            gripper_width=qpos[6] + qpos[7],
            phase=frame.phase,
        )
        return self.observation

    def _metrics(self):
        return self.sim.task.metrics(
            self.sim.task_adapter.task_poses(self.sim.data, self.sim.task))

    def _relation(self):
        grip = self.sim.data.body('right/gripper_center')
        plug = self.sim.data.body('task/yellow_plug')
        rotation = grip.xmat.reshape(3, 3).T
        return (rotation @ (plug.xpos - grip.xpos)).copy()

    def _physics_interval(self, allow_force):
        interval_penetration = 0.
        completed = True
        for _ in range(self.sim.decimation * self.sim.substeps):
            self.sim.data.xfrc_applied[self.plug_body] = 0.
            forcing = allow_force and self.force_remaining > 0
            if forcing:
                rotation = self.sim.data.body('right/gripper_center').xmat.reshape(3, 3)
                self.sim.data.xfrc_applied[self.plug_body, :3] = (
                    rotation @ self.requested_direction * self.condition['force_n'])
            mujoco.mj_step(self.sim.model, self.sim.data)
            mujoco.mj_forward(self.sim.model, self.sim.data)
            if self.sim.data.ncon:
                interval_penetration = max(
                    interval_penetration, -min(contact.dist for contact in self.sim.data.contact))
            if forcing:
                self.force_remaining -= 1
                self.force_substeps_applied += 1
                if self.force_remaining == 0:
                    self.force_end_relation = self._relation()
                    self.settle_remaining = self.settle_steps
            elif self.settle_remaining > 0:
                self.settle_remaining -= 1
                if self.settle_remaining == 0:
                    self.settled_relation = self._relation()
            if interval_penetration > .0002:
                completed = False
                break
        self.sim.data.xfrc_applied[self.plug_body] = 0.
        self.max_penetration = max(self.max_penetration, interval_penetration)
        self.step_penetration = max(self.step_penetration, interval_penetration)
        if completed:
            self.sim._images = None
            self.sim.task.metrics(
                self.sim.task_adapter.task_poses(self.sim.data, self.sim.task),
                self.sim.dt * self.sim.decimation)
        return completed

    def _advance_policy_frame(self, index, target):
        frame = self.player.frames[index]
        if frame.phase == 'release':
            target = self.previous_target
        elif frame.phase == 'retreat':
            target = np.zeros(2)
        previous_position = self.smoother.position.copy()
        previous_velocity = self.smoother.velocity.copy()
        xy = self.smoother.update(target)
        self.last_action = inject_xy(
            frame.nominal_action, xy, self.joint_maps[index],
            self.sim.layout.lower, self.sim.layout.upper)
        self.sim.apply(self.sim.layout.command(self.last_action, self.sim.sim_time, index))
        completed = self._physics_interval(allow_force=True)
        self.index = index
        joint_delta = self.last_action[7:13] - np.asarray(frame.nominal_action[7:13])
        self.correction_integral += float(np.linalg.norm(xy)) / 30
        self.correction_variation += float(np.linalg.norm(xy - previous_position))
        self.max_residual = max(self.max_residual, float(np.max(np.abs(xy))))
        self.max_speed = max(self.max_speed, float(np.linalg.norm(self.smoother.velocity, ord=np.inf)))
        acceleration = (self.smoother.velocity - previous_velocity) * 30
        self.max_acceleration = max(self.max_acceleration, float(np.linalg.norm(acceleration, ord=np.inf)))
        self.max_joint_correction = max(self.max_joint_correction, float(np.max(np.abs(joint_delta))))
        self.previous_target = np.asarray(target).copy()
        if completed:
            self._observe(index)
            if index == self.anchor:
                self.pre_force_relation = self._relation()
                self.force_remaining = self.force_steps
        return completed

    def _run_post_retreat(self):
        for index in range(self.retreat_end + 1, self.episode_end + 1):
            frame = self.player.frames[index]
            action = np.asarray(frame.nominal_action)
            self.sim.apply(self.sim.layout.command(action, self.sim.sim_time, index))
            if not self._physics_interval(allow_force=False):
                self.index = index
                return False
            self.index = index
        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        mujoco.mj_setState(
            self.sim.model, self.sim.data, np.asarray(self.initial['values']), self.initial['spec'])
        self.sim.task.hold = 0.
        mujoco.mj_forward(self.sim.model, self.sim.data)
        explicit = options.get('condition') if options else None
        self.condition = explicit or sample_contact_drift(
            self.design['distribution'], self.np_random, f'train_{self.episode_number:08d}')
        self.episode_number += 1
        self.requested_direction = np.array([
            np.cos(self.condition['angle_rad']), np.sin(self.condition['angle_rad']), 0.])
        self.force_steps = int(round(
            self.condition['duration_ms'] / 1000 / self.sim.model.opt.timestep))
        self.settle_steps = int(round(.2 / self.sim.model.opt.timestep))
        self.force_remaining = self.settle_remaining = self.force_substeps_applied = 0
        self.pre_force_relation = self.force_end_relation = self.settled_relation = None
        self.index = self.start
        self.history = FullTaskProprioHistory(15)
        self.smoother = SmoothResidual()
        self.last_action = np.asarray(self.player.frames[self.start].nominal_action).copy()
        self.previous_target = np.zeros(2)
        self.previous_metrics = self._metrics()
        self.correction_integral = self.correction_variation = 0.
        self.max_penetration = self.max_residual = self.max_speed = 0.
        self.step_penetration = 0.
        self.max_acceleration = self.max_joint_correction = 0.
        observation = self._observe(self.start + 1)
        return observation, {}

    def step(self, action):
        target = np.clip(np.asarray(action, dtype=float), -1., 1.)
        self.step_penetration = 0.
        action_delta = float(np.sum((target - self.previous_target) ** 2))
        collision = False
        for _ in range(2):
            if self.index == self.retreat_end:
                break
            if not self._advance_policy_frame(self.index + 1, target):
                collision = True
                break
        finished = self.index == self.retreat_end and not collision
        if finished:
            collision = not self._run_post_retreat()
        current = self._metrics()
        success = bool(finished and not collision and self.index == self.episode_end and current['success'])
        reward = (50. if success else 0.) - (20. if collision else 0.) - .01
        reward += 500 * (self.previous_metrics['face_gap_m'] - current['face_gap_m'])
        reward += 2000 * (self.previous_metrics['lateral_error_m'] - current['lateral_error_m'])
        reward -= .1 * self.step_penetration / .0002
        reward -= .03 * float(np.sum((self.smoother.position / self.smoother.bound) ** 2))
        reward -= .01 * action_delta
        self.previous_metrics = current
        done = collision or finished
        info = {}
        if done:
            info = {
                'is_success': success,
                'collision': bool(collision),
                'condition': self.condition,
                'force_substeps_applied': self.force_substeps_applied,
                'max_penetration_m': self.max_penetration,
                'correction_integral_m_s': self.correction_integral,
                'correction_variation_m': self.correction_variation,
                'max_residual_m': self.max_residual,
                'max_residual_speed_m_s': self.max_speed,
                'max_residual_acceleration_m_s2': self.max_acceleration,
                'max_joint_correction_rad': self.max_joint_correction,
                'final_metrics': current,
                'privileged_student_inputs': 0,
                'student_observation': 'P0 15-frame right proprioception/nominal/feedback/gripper/phase history',
                'nominal_reference_hash': self.player.reference_hash,
                'design_sha256': self.design_sha256,
            }
        return self.observation, float(reward), bool(done), False, info

    def close(self):
        self.sim.close()

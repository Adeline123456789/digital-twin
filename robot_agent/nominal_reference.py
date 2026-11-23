"""Immutable SI commands; the player has no simulator, planner or object state."""
from dataclasses import dataclass
import hashlib
import json
import math


ACTION_NAMES = tuple(f'{arm}.{name}' for arm in ('left', 'right')
                     for name in ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper_width'))


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ReferenceFrame:
    timestep: float
    phase: str
    nominal_action: tuple
    nominal_joint_target: tuple


class ReferencePlayer:
    def __init__(self, payload, expected_hash):
        if content_hash(payload) != expected_hash:
            raise ValueError('Frozen reference hash mismatch')
        if payload['action_names'] != list(ACTION_NAMES) or payload['action_units'] != ['rad'] * 6 + ['m'] + ['rad'] * 6 + ['m']:
            raise ValueError('Expected dual-arm SI position targets')
        frames = tuple(ReferenceFrame(float(r['timestep']), r['phase'], tuple(r['nominal_action']),
                                      tuple(r['nominal_joint_target'])) for r in payload['frames'])
        previous = 0.
        for frame in frames:
            if (len(frame.nominal_action) != 14 or len(frame.nominal_joint_target) != 12
                    or not all(math.isfinite(v) for v in (*frame.nominal_action, *frame.nominal_joint_target, frame.timestep))
                    or not 0 < frame.timestep - previous <= .034):
                raise ValueError('Invalid action or discontinuous 30 Hz reference')
            if frame.nominal_joint_target != frame.nominal_action[:6] + frame.nominal_action[7:13]:
                raise ValueError('Joint target disagrees with position action')
            previous = frame.timestep
        if not frames:
            raise ValueError('Empty reference')
        self.frames = frames
        self.reference_hash = expected_hash
        self.nominal_action_hash = content_hash([f.nominal_action for f in frames])

    def command(self, index):
        return self.frames[index]


class PlannerSpy:
    """Intercept the audited native planner entry during frozen execution."""
    def __init__(self):
        self.calls = 0

    def forbidden(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError('Planner called after reference freeze')


class PrivilegedAccessAudit:
    """Record rejected policy input requests; never retain privileged values."""
    def __init__(self):
        self.policy_reads = 0
        self.accesses = dict(goal=0, object=0, contact=0, other=0)

    def reject(self, field):
        self.policy_reads += 1
        category = next((name for name in ('goal', 'object', 'contact') if name in field), 'other')
        self.accesses[category] += 1
        raise PermissionError('Privileged policy input denied: ' + field)

    def record(self):
        return dict(privileged_policy_reads=self.policy_reads, privileged_access_attempts=dict(self.accesses),
                    isolation='ReferencePlayer receives only immutable SI actions and phase/time; no backend or MjData',
                    limit='Interface/code audit, not an operating-system security sandbox')


def freeze_payload(rows, identity):
    return dict(schema_version=1, **identity, action_names=list(ACTION_NAMES),
                action_units=['rad'] * 6 + ['m'] + ['rad'] * 6 + ['m'],
                action_semantics='native joint-position targets and full gripper width; post-step sample times',
                frames=[dict(timestep=r['time'], phase=r['phase'], nominal_action=r['action'],
                             nominal_joint_target=r['action'][:6] + r['action'][7:13]) for r in rows])

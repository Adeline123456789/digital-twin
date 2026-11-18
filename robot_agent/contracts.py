"""Facts in a single clock/frame. Null means not measured."""
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import PurePath
import re


@dataclass(frozen=True)
class Reference:
    root: str
    path: str
    sha256: str

    def __post_init__(self):
        path = PurePath(self.path)
        if self.root not in {'dataset', 'repository'} or path.is_absolute() or '..' in path.parts:
            raise ValueError('Use a scoped relative evidence path')
        if not re.fullmatch('[0-9a-f]{64}', self.sha256):
            raise ValueError('Missing evidence digest')


@dataclass(frozen=True)
class Observation:
    experiment_id: str
    source_profile: str
    native_source_hash: str
    world_model_version: str
    policy_version: str
    task_id: str
    target_index: int
    phase: str
    action_id: str
    timestamp_mono: float | None
    clock_source: str
    time_alignment_quality: str
    frame_id: str
    units: str
    sensor_provenance: str
    observability: str
    observed_value: object
    evidence_refs: tuple
    predicted_value: object = None
    residual: object = None
    uncertainty: float | None = None

    def issue(self):
        if not self.experiment_id or not self.task_id or not self.policy_version or not self.world_model_version or not re.fullmatch('[0-9a-f]{64}', self.native_source_hash):
            return 'VERSION_UNKNOWN'
        real = self.source_profile == 'real'
        if self.source_profile not in {'real', 'original', 'retained_final'}:
            return 'VERSION_UNKNOWN'
        clocks = {'recorded_monotonic', 'camera_exposure'} if real else {'sim_time'}
        if self.time_alignment_quality != 'verified' or self.clock_source not in clocks or self.timestamp_mono is None or not isfinite(self.timestamp_mono):
            return 'TIMESTAMP_MISMATCH'
        sensors = {
            'real.rgb': ('image', 'px'), 'real.gripper': ('gripper', 'mm'),
            'real.proprio': ('joint', 'deg'), 'sim.privileged_contact': ('world', 'm'),
            'sim.object_pose': ('world', 'm'), 'sim.socket_residual': ('socket', 'mm'),
            'sim.native_report': ('simulator', 'status')}
        if self.sensor_provenance.startswith('real.') != real or self.sensor_provenance not in sensors:
            return 'SENSOR_UNAVAILABLE'
        if (self.frame_id, self.units) != sensors[self.sensor_provenance]:
            return 'FRAME_OR_UNITS_MISMATCH'
        if not self.evidence_refs or self.observed_value is None or self.observability not in {'observed','estimated'}:
            return 'SENSOR_UNAVAILABLE'
        if self.sensor_provenance == 'real.rgb' and (self.observability != 'estimated' or self.uncertainty is None or not isfinite(self.uncertainty) or self.uncertainty < 0):
            return 'SENSOR_UNAVAILABLE'
        return None


@dataclass(frozen=True)
class Event(Observation):
    kind: str = 'SENSOR_UNAVAILABLE'
    severity: str = 'warning'
    status: str = 'UNKNOWN'
    first_observed_frame: int | None = None
    first_observed_time: float | None = None
    confirmation_time: float | None = None
    hypothesis_ids: tuple = ()
    root_cause_status: str = 'UNKNOWN'


@dataclass
class FailureReport:
    experiment_id: str
    events: list
    first_divergence: Event | None
    status: str
    task_success: bool | None
    evidence_refs: tuple = ()
    physical_onset: float | None = None
    root_cause_status: str = 'UNKNOWN'

    def to_dict(self):
        return asdict(self)

"""Interaction evidence with explicit UNKNOWN and Strong Kleene logic."""
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from .contracts import Reference


class Truth(str, Enum):
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    UNKNOWN = 'UNKNOWN'

    def __and__(self, other):
        if self is Truth.FALSE or other is Truth.FALSE:
            return Truth.FALSE
        return Truth.TRUE if self is other is Truth.TRUE else Truth.UNKNOWN

    def __or__(self, other):
        if self is Truth.TRUE or other is Truth.TRUE:
            return Truth.TRUE
        return Truth.FALSE if self is other is Truth.FALSE else Truth.UNKNOWN

    def __invert__(self):
        return {Truth.TRUE: Truth.FALSE, Truth.FALSE: Truth.TRUE, Truth.UNKNOWN: Truth.UNKNOWN}[self]


class Comparison(str, Enum):
    MATCHED = 'MATCHED'
    PENDING = 'PENDING'
    CONTRADICTED = 'CONTRADICTED'
    UNRESOLVED = 'UNRESOLVED'
    UNEXPECTED = 'UNEXPECTED'


EVENT_SEQUENCE = ('APPROACH_REACHED', 'GRASP_ESTABLISHED', 'LIFT_CONFIRMED', 'RETENTION_OK',
                  'PREINSERT_REACHED', 'ALIGNMENT_OK', 'INSERTION_PROGRESS', 'SEATED', 'RELEASE_STABLE')

EVENT_MODALITIES = {
    'TARGET_LOCKED': 'DIRECT', 'LOCAL_VISUAL_ALIGNMENT_AVAILABLE': 'DIRECT',
    'GRIPPER_STATE_CHANGED': 'DIRECT', 'IN_HAND_POSE_DRIFT_LIKELY': 'INFERRED',
    'CONTACT_MODE_LIKELY': 'INFERRED', 'IN_HAND_POSE_DRIFT': 'PRIVILEGED_EVALUATOR_ONLY',
    'EXACT_CONTACT_PAIR': 'PRIVILEGED_EVALUATOR_ONLY', 'EXACT_INSERTION_DEPTH': 'PRIVILEGED_EVALUATOR_ONLY',
}


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    event_name: str
    phase: str
    expected_presence: bool
    earliest_time: float
    deadline: float
    prerequisites: tuple = ()
    observation_requirements: tuple = ()

    def __post_init__(self):
        if not (isfinite(self.earliest_time) and isfinite(self.deadline) and 0 <= self.earliest_time <= self.deadline):
            raise ValueError('Invalid event time window')


@dataclass(frozen=True)
class ObservedEvent:
    event_id: str
    status: Truth
    confidence: float
    timestamp: float
    evidence_ref: Reference | None
    available_observations: tuple = ()
    clock_valid: bool = True


@dataclass(frozen=True)
class EventComparison:
    spec: EventSpec
    observation: ObservedEvent | None
    status: Comparison
    timestamp: float
    reason: str


@dataclass
class InteractionEventTrace:
    comparisons: list = field(default_factory=list)
    matched: set = field(default_factory=set)
    last_time: float | None = None
    clock_valid: bool = True
    missing_in_window: set = field(default_factory=set)

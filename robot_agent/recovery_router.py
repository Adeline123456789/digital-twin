"""Evidence-only routes. No execution, language model or Twin writer."""
from enum import Enum
from .interaction_events import Comparison as C


class Route(str, Enum):
    NO_ACTION = 'NO_ACTION'
    RETARGET = 'RETARGET'
    RESIDUAL = 'RESIDUAL'
    STRUCTURAL_CANDIDATE = 'STRUCTURAL_CANDIDATE'
    UNKNOWN = 'UNKNOWN'


def route(comparison, *, validated_recoveries=(), within_bounds=False,
          static_geometry_verified=False, repeated_structural_evidence=False):
    if comparison.status in (C.MATCHED, C.PENDING, C.UNRESOLVED):
        return Route.NO_ACTION
    if static_geometry_verified and within_bounds:
        return Route.RETARGET
    if comparison.spec.event_name in validated_recoveries and within_bounds:
        return Route.RESIDUAL
    if repeated_structural_evidence:
        return Route.STRUCTURAL_CANDIDATE
    return Route.UNKNOWN


def safety_decision(comparison, critical=False):
    return 'HOLD' if critical and comparison.status is C.UNRESOLVED else 'CONTINUE'

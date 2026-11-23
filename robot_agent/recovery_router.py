"""Evidence-only routes. No execution, language model or Twin writer."""
from enum import Enum
from .interaction_events import Comparison as C


class Route(str, Enum):
    NO_ACTION = 'NO_ACTION'
    RETARGET = 'RETARGET'
    RESIDUAL = 'RESIDUAL'
    REACQUIRE_OBSERVATION = 'REACQUIRE_OBSERVATION'
    STRUCTURAL_CANDIDATE = 'STRUCTURAL_CANDIDATE'
    UNKNOWN = 'UNKNOWN'


def route(comparison, *, validated_recoveries=(), within_bounds=False,
          static_geometry_verified=False, repeated_structural_evidence=False,
          safe_precontact=False, contact_likely=False):
    if comparison.status is C.UNRESOLVED and safe_precontact and not contact_likely:
        return Route.REACQUIRE_OBSERVATION
    if comparison.status in (C.MATCHED, C.PENDING, C.UNRESOLVED):
        return Route.NO_ACTION
    if static_geometry_verified and within_bounds:
        return Route.RETARGET
    if comparison.spec.event_name in validated_recoveries and within_bounds:
        return Route.RESIDUAL
    if repeated_structural_evidence:
        return Route.STRUCTURAL_CANDIDATE
    return Route.UNKNOWN


def observation_mode(comparison, *, contact_likely=False):
    return ('PROPRIOCEPTIVE_CONTACT_MODE' if comparison.status is C.UNRESOLVED and contact_likely
            else 'RGB_PROPRIOCEPTION')


def safety_decision(comparison, critical=False):
    return 'HOLD' if critical and comparison.status is C.UNRESOLVED else 'CONTINUE'

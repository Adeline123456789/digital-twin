"""Unvalidated or missing RGB features propagate UNKNOWN into interaction predicates."""
from .interaction_events import ObservedEvent, Truth as T


def alignment_event(observation, evidence_ref, tolerance_px=3.):
    visual = observation.visual
    status = T.UNKNOWN
    if visual.confidence >= .9 and visual.relative_image_residual is not None:
        status = T.TRUE if sum(value ** 2 for value in visual.relative_image_residual) <= tolerance_px ** 2 else T.FALSE
    return ObservedEvent('ALIGNMENT_OK', status, visual.confidence, observation.timestamp, evidence_ref,
                         ('rgb', 'plug_tip', 'socket_identity') if status is not T.UNKNOWN else ('rgb',))

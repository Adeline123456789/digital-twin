from robot_agent.contracts import Reference
from robot_agent.interaction_events import EventSpec, ObservedEvent, Truth as T, Comparison as C
from robot_agent.event_comparator import compare_event

REF = Reference('repository', 'tests/event_fixture', 'a' * 64)

def spec(presence=True):
    return EventSpec('alignment', 'ALIGNMENT_OK', 'align', presence, 1., 2., (), ('rgb',))

def observed(status, time=1.5, confidence=1.):
    return ObservedEvent('alignment', status, confidence, time, REF, ('rgb',))

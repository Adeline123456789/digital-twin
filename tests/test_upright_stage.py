import math
from validate_shared_reference import upright_stage_check


def test_retained_but_tilted_plug_is_not_upright():
    def row(phase, degrees):
        return dict(phase=phase, task_metrics=dict(tilt_rad=math.radians(degrees)))

    assert not upright_stage_check([row('lift', 2.963)])['passed']
    assert upright_stage_check([row('lift', .44), row('fold', 65)])['passed']
    assert not upright_stage_check([row('fold', .1)])['passed']
    assert not upright_stage_check([row('lift', float('nan'))])['passed']

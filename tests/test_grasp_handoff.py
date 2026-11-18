from grasp_handoff import screen_lift, yellow_top


def test_rejects_known_lift_rotation_and_missing_observation():
    assert not screen_lift(44.3, 33.6, 285, 267)['allow_transport']
    assert not screen_lift(41., 40., 285, 260)['allow_transport']
    assert not screen_lift(41., 40., None, 285)['allow_transport']
    assert not screen_lift(41., float('nan'), 285, 285)['allow_transport']
    assert screen_lift(43.1, 41., 285, 282)['allow_transport']


def test_horizontal_grasp_top_is_not_clipped_at_old_vertical_roi():
    import numpy as np
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[230:300, 260:380] = [0, 255, 255]
    assert yellow_top(image) == 230
    image[:] = 0
    assert yellow_top(image) is None

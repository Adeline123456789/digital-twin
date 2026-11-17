from grasp_handoff import screen_lift


def test_rejects_known_lift_rotation_and_missing_observation():
    assert not screen_lift(44.3, 33.6, 285, 267)['allow_transport']
    assert not screen_lift(41., 40., 285, 260)['allow_transport']
    assert not screen_lift(41., 40., None, 285)['allow_transport']
    assert not screen_lift(41., float('nan'), 285, 285)['allow_transport']
    assert screen_lift(43.1, 41., 285, 282)['allow_transport']

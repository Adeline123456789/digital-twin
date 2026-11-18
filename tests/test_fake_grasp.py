from robot_agent.fake_grasp import attachment_flags,check_grasp


def test_active_weld_mocap_and_missing_history():
    import mujoco
    xml='<mujoco><worldbody><body name="task/yellow_plug"><freejoint/><geom size=".01"/></body></worldbody><equality><weld body1="task/yellow_plug"/></equality></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    assert check_grasp(attachment_flags(model,data),[])=='REJECT_AND_ROLLBACK'
    data.eq_active[0]=0
    assert check_grasp(attachment_flags(model,data),[])=='DEFER_UNKNOWN'
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body mocap="true" name="task/yellow_plug"><geom size=".01"/></body></worldbody></mujoco>')
    assert 'mocap-driven object' in attachment_flags(model,mujoco.MjData(model))


def test_disconnected_follow_and_direct_pose_edits_reject():
    samples=[dict(phase='lift',open=True,unsupported=True,plug=[0,0,i*.01],grip=[0,0,.02+i*.01]) for i in range(3)]
    assert check_grasp([],samples)=='REJECT_AND_ROLLBACK'
    assert check_grasp([],[],direct_pose_write=True)=='REJECT_AND_ROLLBACK'
    assert check_grasp([],samples[:1])=='DEFER_UNKNOWN'


def test_native_active_inactive_gap_adapter():
    import mujoco
    from dataclasses import replace
    from robot_agent.contact_probe import contact_events,inspect_state
    from test_observation_contract import OBS
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1" margin=".02" gap=".02"/><body name="task/yellow_plug" pos="0 0 .11"><freejoint/><geom name="plug" size=".1" margin=".02" gap=".02"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model)
    o=replace(OBS,frame_id='world',units='m',sensor_provenance='sim.privileged_contact')
    for height,expected in [(.11,False),(.099,True)]:
        data.qpos[2]=height
        mujoco.mj_forward(model,data)
        state=inspect_state(model,data,surfaces=False)
        assert state['contacts'][0]['solver_active'] is expected
        events=contact_events(o,state)
        assert any(e.kind=='CONTACT_INACTIVE' for e in events) is (not expected)
        assert any(e.kind=='COLLISION_GAP' for e in events)
        assert all(e.severity=='info' for e in events)

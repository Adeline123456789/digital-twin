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

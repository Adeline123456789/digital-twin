import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from task_relative import (RelativeTask, SCENE, body_pose, outlet_frame, forbidden_contact,
                           read_stopped_rgb, grasp_drift, pin_points)
from validate_shared_reference import MujocoBackend, load
import yaml


class RightPrecontactTests(unittest.TestCase):
    def test_precontact_obstacles(self):
        def pair(a,b):
            return dict(body1=a,body2=b)
        self.assertIsNone(forbidden_contact([pair('task/black_charger','task/board')], True))
        for a,b in [('task/black_charger','right/link7'),('task/yellow_plug','task/strip')]:
            contact=pair(a,b)
            self.assertEqual(forbidden_contact([contact],True), contact)

    def test_stopped_rgb_cannot_borrow_a_simulated_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            cal=dict(camera_id='stationary',max_pair_skew_s=.01)
            (directory/'cal.json').write_text(json.dumps(cal))
            bundle=dict(arm='right',target_index=3,tool_frame='right/gripper_center',
                        world_frame='simulation_world',camera_id='stationary',phase='precontact',
                        clock='recorded_monotonic',calibration='cal.json',fk_time_s=1.)
            path=directory/'sample.json'
            path.write_text(json.dumps(bundle))
            obs=dict(valid=True,source='synthetic_fiducial_assisted',timestamp_s=1.)
            with patch('relative_pose.measure_rgb',return_value=obs):
                self.assertFalse(read_stopped_rgb(path)['valid'])
                obs.update(source='fiducial_assisted',timestamp_s=1.5)
                self.assertIn('pairing',read_stopped_rgb(path)['reason'])

    def test_initialization_uses_measured_socket_and_grasp_once(self):
        # Synthetic boundary values in an actual right-only model. No hardware or camera input.
        sim=MujocoBackend(dict(load(),task_scene=yaml.safe_load(SCENE.read_text())))
        try:
            with tempfile.TemporaryDirectory() as folder:
                wg=body_pose(sim,'right/gripper_center')
                ws=body_pose(sim,'task/strip') @ outlet_frame(sim)
                ws[0,3]+=.001
                gp=np.linalg.solve(wg,body_pose(sim,'task/yellow_plug'))
                gp[1,3]+=.001
                obs=dict(T_WS=ws.tolist(),T_GP=gp.tolist(),T_WG=wg.tolist(),
                    pin_tips_P_m=[[0,0,0],[.01,0,0]],insertion_axis_P=[0,0,1],
                    right_state_si=sim.read().position.tolist(),
                    quality={'task_tolerance':{'max_tip_error_m':.001,'max_axis_error_deg':1}})
                task=RelativeTask(Path(folder),dict(stage='precontact',measurement=obs))
                task.restore_held(sim)
                np.testing.assert_allclose(body_pose(sim,'task/strip') @ outlet_frame(sim), ws, atol=1e-10)
                np.testing.assert_allclose(body_pose(sim,'task/yellow_plug'), wg @ gp, atol=1e-10)
                self.assertLess(grasp_drift(sim,gp,pin_points(sim)),1e-9)
                self.assertTrue((Path(folder)/'initial_strip.npz').is_file())
                wrong=gp.copy()
                wrong[1,3]+=.002
                task.grasp,task.points=wrong,pin_points(sim)
                with self.assertRaisesRegex(ValueError,'grasp changed'):
                    task.check_step(sim,'align',dict(contacts=[]))
        finally:
            sim.close()

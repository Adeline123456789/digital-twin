import json
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def transform(angles, position):
    t = np.eye(4); t[:3,:3] = Rotation.from_euler('xyz',angles,degrees=True).as_matrix(); t[:3,3] = position
    return t


def synthetic_observations(single_axis=False):
    rng = np.random.default_rng(19)
    mount = transform([8,-5,12],[.04,.01,.03])
    fixed_target = transform([0,0,0],[0,0,.65])
    points = np.array([[x,y,0] for x in [-.08,0,.08] for y in [-.03,.03]],float)
    k = np.array([[600,0,320],[0,600,240],[0,0,1]],float)
    g=[]; c=[]
    for i in range(10):
        pose = transform([0,0,i*3] if single_axis else rng.uniform(-18,18,3),rng.uniform(-.05,.05,3))
        camera = np.linalg.inv(pose@mount)@fixed_target
        pixels,_ = cv2.projectPoints(points,Rotation.from_matrix(camera[:3,:3]).as_rotvec(),camera[:3,3],k,None)
        ok,r,t = cv2.solvePnP(points,pixels,k,None,flags=cv2.SOLVEPNP_ITERATIVE)
        assert ok
        estimated = np.eye(4); estimated[:3,:3] = cv2.Rodrigues(r)[0]; estimated[:3,3] = t.ravel()
        g.append(pose.tolist()); c.append(estimated.tolist())
    baseline=mount@transform([1,-1,2],[.01,-.005,.003])
    return dict(translation_unit='m',T_base_gripper=g,T_camera_target=c,
                baseline_T_gripper_camera=baseline.tolist(),split=['train']*6+['heldout']*4,
                bounds=dict(max_translation_change_m=.05,max_rotation_change_deg=20),
                provenance='SYNTHETIC projected RGB correspondences; known metric target; not a real measurement'),mount


def run_adapter(data):
    # The installed MuJoCo OpenCV 5 build lacks this API; system OpenCV has it.
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/'poses.json'; source.write_text(json.dumps(data))
        output=Path(tmp)/'result'
        proc=subprocess.run(['/usr/bin/python3',str(ROOT/'scripts/solve_rgb_handeye.py'),
                             '--input',str(source),'--output',str(output)],text=True,capture_output=True)
        return proc.returncode,json.loads((output/'result.json').read_text())


class RGBHandEyeTests(unittest.TestCase):
    def test_missing_input_still_records_infrastructure_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result'
            proc=subprocess.run(['/usr/bin/python3',str(ROOT/'scripts/solve_rgb_handeye.py'),
                                 '--input',str(Path(tmp)/'missing.json'),'--output',str(output)],capture_output=True)
            report=json.loads((output/'result.json').read_text())
            self.assertEqual(proc.returncode,2)
            self.assertEqual(report['result'],'infrastructure_error')
            self.assertIn('FileNotFoundError',report['error'])

    def test_capture_feedback_and_exposure_match_existing_mujoco_fk(self):
        import mujoco
        sys.path.insert(0,str(ROOT/'scripts'))
        from prepare_rgb_handeye import robot_observations, SIM
        end = np.array([10,20,-30,40,-10,60],float)
        with tempfile.TemporaryDirectory() as tmp:
            capture=Path(tmp);(capture/'right_wrist').mkdir()
            (capture/'right_wrist/camera.json').write_text(json.dumps({'samples':[
                dict(timestamp_domain='timestamp_domain.global_time',host_monotonic=100,
                     host_unix=1000,rgb_device_ms=1000500)]}))
            with (capture/'commanded_probe.csv').open('w') as f:
                writer=csv.writer(f);writer.writerow(['monotonic','phase']+[f'feedback_j{i}' for i in range(1,7)]+[f'target_j{i}' for i in range(1,7)])
                writer.writerow([100,'begin',*np.zeros(6),*np.full(6,90)])
                writer.writerow([101,'end',*end,*np.full(6,90)])
            observation=robot_observations(capture,[0])
        model=mujoco.MjModel.from_xml_path(str(SIM/'assets/mujoco/workcell.xml'));state=mujoco.MjData(model)
        for i,q in enumerate(np.deg2rad(end/2)):
            state.qpos[model.joint(f'right/joint{i+1}').qposadr[0]]=q
        mujoco.mj_forward(model,state)
        transforms=[]
        for name in ['right/base_link','right/link6']:
            t=np.eye(4);body=state.body(name);t[:3,:3]=body.xmat.reshape(3,3);t[:3,3]=body.xpos;transforms.append(t)
        np.testing.assert_allclose(observation['T_base_gripper'][0],np.linalg.inv(transforms[0])@transforms[1],atol=1e-8)
        self.assertEqual(observation['exposure_monotonic_s'],[100.5])
        self.assertEqual(observation['phase'],['begin'])

    def test_pnp_to_handeye_and_heldout_not_used_to_fit(self):
        data,true_mount=synthetic_observations()
        code,report=run_adapter(data)
        self.assertEqual(code,0,report)
        np.testing.assert_allclose(report['T_gripper_camera'],true_mount,atol=1e-6)
        self.assertLess(report['candidate']['heldout']['translation_rmse_m'],1e-6)
        self.assertFalse(report['calibration_promoted'])
        # Move only the held-out target. The fit must stay exactly unchanged.
        for i in range(6,10):data['T_camera_target'][i][0][3]+=.02
        code,shifted=run_adapter(data)
        self.assertEqual(code,0,shifted)
        np.testing.assert_array_equal(shifted['T_gripper_camera'],report['T_gripper_camera'])
        self.assertGreater(shifted['candidate']['heldout']['translation_rmse_m'],.019)

    def test_single_axis_and_wrong_units_are_rejected(self):
        data,_=synthetic_observations(single_axis=True)
        code,report=run_adapter(data)
        self.assertEqual(code,2); self.assertIn('unobservable',report['error'])
        self.assertNotIn('T_gripper_camera',report)
        data,_=synthetic_observations();data['translation_unit']='mm'
        code,report=run_adapter(data)
        self.assertEqual(code,2);self.assertIn('meter-unit',report['error'])

    def test_requested_bound_is_not_relaxed(self):
        data,_=synthetic_observations();data['bounds']['max_translation_change_m']=.001
        code,report=run_adapter(data)
        self.assertEqual(code,2)
        self.assertEqual(report['validation_status'],'REJECTED_OUTSIDE_REQUESTED_BOUNDS')
        self.assertFalse(report['calibration_promoted'])

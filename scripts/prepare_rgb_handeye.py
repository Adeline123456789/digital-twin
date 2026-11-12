"""Read existing right-wrist captures into metric robot poses; no calibration fit.

Uses the separate Piper project's URDF FK, measured joints and RGB exposure
clock. Output is a local calibration artifact, not a new Dataset format.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from learn import SIM, digest, save

sys.path.insert(0,str(SIM/'src'))
from piper_sim.kinematics import forward


def robot_observations(capture, frames):
    camera = capture/'right_wrist/camera.json'; command = capture/'commanded_probe.csv'
    urdf = SIM/'assets/mujoco/kinematics.urdf'
    samples = json.loads(camera.read_text())['samples']
    selected = [samples[i] for i in frames]
    if any(s['timestamp_domain'] != 'timestamp_domain.global_time' for s in selected):
        raise ValueError('RGB samples must have the exporter-supported global exposure clock')
    exposure = np.array([s['host_monotonic']+(s['rgb_device_ms']/1000-s['host_unix']) for s in selected])
    rows = list(csv.DictReader(command.open()))
    times = np.array([float(r['monotonic']) for r in rows])
    if np.any(np.diff(times)<=0) or np.any(exposure<times[0]) or np.any(exposure>times[-1]):
        raise ValueError('Exposure must be inside the ordered feedback interval; no extrapolation')
    measured = np.array([[np.interp(t,times,[float(r[f'feedback_j{i}']) for r in rows])
                         for i in range(1,7)] for t in exposure])
    poses = [forward(urdf,{f'joint{i+1}':q for i,q in enumerate(np.deg2rad(angles))})['link6']
             for angles in measured]
    return dict(translation_unit='m', T_base_gripper=np.array(poses).tolist(), frame_indices=frames,
                exposure_monotonic_s=exposure.tolist(),
                phase=[rows[i]['phase'] for i in np.searchsorted(times,exposure,side='right')-1],
                status='ROBOT_POSES_ONLY_RGB_TARGET_POSES_MISSING',
                provenance=dict(source_capture=str(capture), robot_frame='URDF base_link', gripper_frame='link6',
                    camera_frame='right wrist RGB optical frame (visual poses not supplied)',
                    pose_source='Existing Piper URDF FK of measured joint feedback; not external end-effector metrology',
                    input_sha256={str(p):digest(p) for p in (camera,command,urdf)},
                    fk_source_sha256=digest(SIM/'src/piper_sim/kinematics.py'),
                    limitations=['Kinematic model and RGB calibration still require independent validation',
                                 'CSV feedback timestamps are host observations, not synchronized encoder exposure timestamps']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--frames',type=int,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    save(args.output,robot_observations(args.capture,args.frames))


if __name__=='__main__':main()

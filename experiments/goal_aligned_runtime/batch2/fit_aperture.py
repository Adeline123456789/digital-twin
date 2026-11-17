"""Offline bounded encoder-width image check using the existing finger silhouette method."""
import csv
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from learn import DATA, digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mount', type=Path)
parser.add_argument('--name', default='aperture_fit')
args = parser.parse_args()
out = Path(__file__).parent
local = DATA/'self_evolution/goal_aligned_batch2'/args.name
local.mkdir(exist_ok=False)
specs = [(1, 'descend', 'train'), (1, 'lift', 'train'),
         (2, 'lift', 'heldout'), (2, 'align', 'heldout')]
model_path = DATA/'complete_sim/succeed/20261117T071047Z_e43b1ad6/proprio_stats/model.xml'
model = mujoco.MjModel.from_xml_path(str(model_path))
if args.mount:
    mount = np.array(json.loads(args.mount.read_text())['T_parent_rgb'])
    camera_id = model.camera('right_wrist').id
    model.cam_pos[camera_id] = mount[:3, 3]
    mujoco.mju_mat2Quat(model.cam_quat[camera_id], (mount[:3,:3] @ np.diag([1,-1,-1])).ravel())
model.vis.quality.offsamples = 0
finger_ids = [i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name in ('right/link7', 'right/link8')]
yy, xx = np.indices((480, 640))
roi = (yy >= 280) & ((xx < 240) | (xx > 440))
samples = []
for number, phase, split in specs:
    capture = DATA/'self_evolution/goal_aligned_batch2'/f'real_{number:06d}'
    rows = list(csv.DictReader((capture/'commanded_probe.csv').open()))
    row = [r for r in rows if r['phase'] == phase][-1]
    camera = json.loads((capture/'right_wrist/camera.json').read_text())['samples']
    frame = min(range(len(camera)), key=lambda i: abs(camera[i]['host_monotonic']-float(row['monotonic'])))
    image = capture/'right_wrist'/f'{frame:04d}.jpg'
    data = mujoco.MjData(model)
    for j in range(1, 7):
        data.qpos[model.joint(f'right/joint{j}').qposadr[0]] = np.deg2rad(float(row[f'feedback_j{j}']))
    samples.append(dict(data=data, mask=cv2.imread(str(image), 0)<45,
        width=float(row['feedback_gripper_open'])*.095, number=number, phase=phase,
        split=split, frame=frame, image_sha256=digest(image)))

started = time.monotonic()
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.enable_segmentation_rendering()
def score(offset, sample, save=False):
    data = sample['data']
    for j in (7, 8):
        data.qpos[model.joint(f'right/joint{j}').qposadr[0]] = (sample['width']+offset/1000)/2
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera='right_wrist')
    image = renderer.render()
    prediction = np.isin(image[:,:,0], finger_ids) & (image[:,:,1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    a, b = prediction & roi, sample['mask'] & roi
    if save:
        cv2.imwrite(str(local/f"{sample['number']}_{sample['phase']}_candidate.png"), prediction.astype('uint8')*255)
    return float((a & b).sum()/(a | b).sum())

try:
    fit = minimize_scalar(lambda offset: -np.mean([score(offset, s) for s in samples if s['split']=='train']),
                          method='bounded', bounds=(-3, 3), options=dict(xatol=.02, maxiter=30))
    comparisons = []
    for sample in samples:
        comparisons.append({**{k:sample[k] for k in ('number','phase','split','frame','image_sha256')},
                            'baseline_iou':score(0, sample), 'candidate_iou':score(fit.x, sample, True)})
    result = dict(candidate_width_offset_mm=float(fit.x), bounds_mm=[-3,3], evaluations=fit.nfev,
        comparisons=comparisons, wall_seconds=time.monotonic()-started, hardware_commands=0,
        model_sha256=digest(model_path), program_sha256=digest(__file__), deployed=False,
        private_mount_sha256=digest(args.mount) if args.mount else None,
        scope='RGB finger silhouette consistency; camera mount and CAD held fixed. Not independent metric calibration.',
        limitations=['Receipt-nearest phase frame; first trial clock uncertainty retained.',
                     'Lighting/occlusion and camera/CAD error may confound width offset.',
                     'No object geometry, contact physics, controller, force or success detector changed.'])
    (out/(args.name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
finally:
    renderer.close()

"""Offline robot-anchored RGB registration in the existing MuJoCo scene.

Adapts EasyHeC's fixed-CAD multi-pose mask objective, not its GPU renderer,
trained segmenter or hardware driver. No physical step or deployed file write.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from learn import DATA, SIM, digest, save


def exposure_states(capture):
    camera = json.loads((capture/'external/camera.json').read_text())
    samples = camera['samples']
    if any(s['timestamp_domain'] != 'timestamp_domain.global_time' for s in samples):
        raise ValueError('Expected existing exporter global exposure clock')
    rows = list(csv.DictReader((capture/'commanded_probe.csv').open()))
    times = np.array([float(r['monotonic']) for r in rows])
    if np.any(np.diff(times) <= 0):
        raise ValueError('Feedback times must increase')
    exposure = np.array([s['host_monotonic']+s['rgb_device_ms']/1000-s['host_unix'] for s in samples])
    values = np.array([[*[float(r[f'feedback_j{i}']) for i in range(1, 7)],
                        float(r['feedback_gripper_open'])*.095] for r in rows])
    q = np.array([np.interp(exposure, times, v) for v in values.T]).T
    window = np.array([np.interp(exposure+.05, times, v)-np.interp(exposure-.05, times, v)
                       for v in values[:, :6].T]).T
    valid = (exposure-.05 >= times[0]) & (exposure+.05 <= times[-1])
    valid &= np.max(abs(window), axis=1) < .1
    phases = np.array([rows[i]['phase'] for i in np.clip(np.searchsorted(times, exposure)-1, 0, len(rows)-1)])
    return camera, exposure, q, valid, phases


def robot_mask(image, threshold):
    # shortcut: this dark robot/bright table ROI excludes the stationary base;
    # use a reviewed segmenter before applying this tool to another scene.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = np.zeros(gray.shape, np.uint8); roi[155:360, 240:640] = 1
    roi[245:290,450:570] = 0  # Fixed holder and its occlusion boundary.
    roi[155:185,610:640] = 0  # Background equipment at the image border.
    mask = ((gray < threshold) & (roi > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    ids = [i for i in range(1, n) if stats[i, 4] > 300]
    if not ids:
        raise ValueError('No visible robot component in frozen ROI')
    mask = labels == max(ids, key=lambda i: stats[i, 4])
    return mask, roi.astype(bool)


def select_views(capture, independent):
    camera, times, q, valid, phases = exposure_states(capture)
    views = []
    for region in ('holder', 'socket'):
        for tag in ('center', 'j4', 'j5', 'j6', 'translation', 'heldout'):
            phase = f'{region}_{tag}_hold'
            indices = np.flatnonzero((phases == phase) & valid)
            if not len(indices):
                raise ValueError('No stationary exposure for '+phase)
            i = int(indices[len(indices)//2])
            views.append(dict(name=phase, split='train' if tag in ('center','j4','j5','j6') else 'heldout',
                              image_path=str(capture/'external'/f'{i:04d}.jpg'), q=q[i].tolist(), frame=i))
    other, times2, q2, valid2, phases2 = exposure_states(independent)
    if camera['serial'] != other['serial'] or not np.allclose(camera['K'], other['K'], atol=1e-6):
        raise ValueError('Independent recording camera identity/intrinsics differ')
    for phase in ('approach', 'descend', 'grasp', 'lift'):
        indices = np.flatnonzero((phases2 == phase) & valid2)
        if len(indices):
            i = int(indices[len(indices)//2])
            views.append(dict(name='independent_'+phase, split='independent',
                              image_path=str(independent/'external'/f'{i:04d}.jpg'), q=q2[i].tolist(), frame=i))
    return camera, views


def changed_pose(baseline, normalized):
    pose = baseline.copy()
    pose[:3, :3] = Rotation.from_rotvec(np.asarray(normalized[:3])*np.deg2rad(5)).as_matrix()@baseline[:3, :3]
    pose[:3, 3] += np.asarray(normalized[3:])*.03
    return pose


def mask_metrics(prediction, reference, roi):
    p, r = prediction & roi, reference & roi
    union = np.count_nonzero(p | r)
    if not union or not p.any() or not r.any():
        raise ValueError('Empty mask cannot validate calibration')
    return dict(iou=float(np.count_nonzero(p & r)/union),
                disagreement=float(np.count_nonzero(p ^ r)/np.count_nonzero(roi)))


def run(args):
    import mujoco
    import yaml
    sys.path.insert(0, str(SIM/'src'))
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.config import load
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(experiment_id=args.output.name, model='deterministic scipy Powell',
                  environment='archived_real_RGB_and_static_simulation_rendering',
                  result='infrastructure_error', root_cause_status='unknown', task_success=None,
                  accepted=False, real_motion_commands=0, physics_steps=0, execution_llm_calls=0,
                  execution_llm_tokens=0, orchestration_tokens=None, privileged_information_used=True,
                  program_sha256=digest(__file__), task_sha256=digest(args.task),
                  threshold=args.threshold, max_evaluations=args.max_evaluations,
                  active_calibration_changed=False)
    sim = renderer = None
    history = []
    try:
        camera, views = select_views(args.capture, args.independent)
        sim = MujocoBackend(dict(load(),task_scene=yaml.safe_load(args.task.read_text())))
        cid = sim.model.camera('overhead').id
        if sim.model.cam_bodyid[cid] != 0:
            raise ValueError('This adapter expects the fixed world camera')
        baseline = np.eye(4); baseline[:3, 3] = sim.model.cam_pos[cid]
        rot = np.empty(9); mujoco.mju_quat2Mat(rot, sim.model.cam_quat[cid])
        baseline[:3, :3] = rot.reshape(3,3)@np.diag([1,-1,-1])
        k = np.array(camera['K'])
        # Identical measured RGB intrinsics for baseline and candidate.
        sim.model.cam_intrinsic[cid] = [k[0,0]/640,k[1,1]/480,(320-k[0,2]-.5)/640,(k[1,2]+.5-240)/480]
        # The installed renderer blends some segmentation IDs with MSAA;
        # native=4/28 errors, offsamples=0: 0/28 at identical tested poses.
        sim.model.vis.quality.offsamples = 0
        renderer = mujoco.Renderer(sim.model, height=240, width=320)
        renderer.enable_segmentation_rendering()
        ids = [g for g in range(sim.model.ngeom)
               if sim.model.body(sim.model.geom_bodyid[g]).name.startswith('right/')]
        states, masks, rois = [], [], []
        for v in views:
            q = np.array(v['q']); sim.set_observation_state(np.r_[np.deg2rad(q[:6]),q[6]])
            d = mujoco.MjData(sim.model)
            d.qpos[:] = sim.data.qpos; d.qvel[:] = sim.data.qvel; d.ctrl[:] = sim.data.ctrl
            mujoco.mj_forward(sim.model,d); states.append(d)
            im = cv2.imread(v['image_path'])
            m, roi = robot_mask(im, args.threshold)
            masks.append(cv2.resize(m.astype(np.uint8),(320,240),interpolation=cv2.INTER_NEAREST)>0)
            rois.append(cv2.resize(roi.astype(np.uint8),(320,240),interpolation=cv2.INTER_NEAREST)>0)
            v['rgb_sha256'] = digest(v['image_path'])
            cv2.imwrite(str(args.output/(v['name']+'_mask.png')),m.astype(np.uint8)*255)
        save(args.output/'views.json',dict(views=views, calibration_source_sha256=digest(args.capture/'external/camera.json')))

        def render(index, pose):
            sim.model.cam_pos[cid] = pose[:3,3]
            mujoco.mju_mat2Quat(sim.model.cam_quat[cid],(pose[:3,:3]@np.diag([1,-1,-1])).ravel())
            mujoco.mj_camlight(sim.model,states[index])
            renderer.update_scene(states[index],camera='overhead')
            seg = renderer.render()
            return np.isin(seg[:,:,0],ids) & (seg[:,:,1] == int(mujoco.mjtObj.mjOBJ_GEOM))

        train = [i for i,v in enumerate(views) if v['split']=='train']
        def objective(x):
            if time.monotonic()-start > args.max_seconds:
                raise TimeoutError('Predeclared wall time budget reached')
            pose = changed_pose(baseline,x)
            loss = np.mean([mask_metrics(render(i,pose),masks[i],rois[i])['disagreement'] for i in train])
            history.append(dict(evaluation=len(history)+1,loss=float(loss),elapsed_s=time.monotonic()-start))
            return loss

        fit = minimize(objective,np.zeros(6),method='Powell',bounds=[(-1,1)]*6,
                       options=dict(maxfev=args.max_evaluations,xtol=.003,ftol=1e-5))
        candidate = changed_pose(baseline,fit.x)
        results = []
        for i,v in enumerate(views):
            row = dict(name=v['name'],split=v['split'],frame=v['frame'])
            im = cv2.imread(v['image_path'])
            for label,pose in [('baseline',baseline),('candidate',candidate)]:
                pred = render(i,pose)
                row[label] = mask_metrics(pred,masks[i],rois[i])
                pred = cv2.resize(pred.astype(np.uint8),(640,480),interpolation=cv2.INTER_NEAREST)>0
                canvas = im.copy(); canvas[pred] = (canvas[pred]*.5+np.array([0,0,255])*.5).astype('uint8')
                cv2.imwrite(str(args.output/(v['name']+'_'+label+'.jpg')),canvas)
            results.append(row)
        save(args.output/'private_calibration.json',dict(T_parent_rgb=candidate.tolist(), baseline_T_parent_rgb=baseline.tolist()))
        report.update(result='success' if fit.success else 'failure', optimizer_success=bool(fit.success),
                      optimizer_message=str(fit.message), evaluations=int(fit.nfev),
                      rotation_change_deg=float(np.linalg.norm(fit.x[:3])*5),
                      translation_change_mm=float(np.linalg.norm(fit.x[3:])*30),
                      at_bound=bool(np.any(abs(fit.x)>.99)), observations=results,
                      private_calibration_sha256=digest(args.output/'private_calibration.json'),
                      view_manifest_sha256=digest(args.output/'views.json'),
                      scope='Pixel fit only. Cable/CAD and dark-mask differences can bias pose; no metric or task-success validation.')
        report['splits'] = {split:{label+'_mean_iou':float(np.mean([r[label]['iou'] for r in results if r['split']==split]))
                                  for label in ('baseline','candidate')}
                            for split in sorted(set(v['split'] for v in views))}
    except Exception as exc:
        report.update(result='infrastructure_error',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        if renderer is not None: renderer.close()
        if sim is not None: sim.close()
        save(args.output/'optimization_trace.json',history)
        report['completed_objective_evaluations'] = len(history)
        report['wall_seconds'] = time.monotonic()-start
        report['local_evidence_path'] = str(args.output.relative_to(DATA)) if args.output.is_relative_to(DATA) else None
        save(args.output/'result.json',report)
        print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('task','capture','independent','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threshold',type=int,choices=(90,110),default=90)
    p.add_argument('--max-evaluations',type=int,default=600)
    p.add_argument('--max-seconds',type=float,default=300)
    run(p.parse_args())

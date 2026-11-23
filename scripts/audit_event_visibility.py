"""Evaluator-only visibility of native pin geometry; never a policy observation."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from learn import DATA, ROOT, digest, save


def audit(ident):
    import mujoco
    import numpy as np
    from contact_evidence import restore
    raw = DATA / 'self_evolution' / ident
    stats = DATA / 'complete_sim/succeed' / ident / 'proprio_stats'
    model = mujoco.MjModel.from_xml_path(str(stats / 'model.xml'))
    data = mujoco.MjData(model)
    groups = {
        'plug_pins': [i for i in range(model.ngeom) if model.geom(i).name.startswith('task/yellow_plug/pin_')],
        'target_panel': [i for i in range(model.ngeom) if model.geom(i).name.startswith('strip/panel_3')],
    }
    if len(groups['plug_pins']) != 3 or not groups['target_panel']:
        raise ValueError('Expected native three pins and outlet 3 panel')
    rows = [json.loads(line) for line in (raw / 'trajectory.jsonl').read_text().splitlines()]
    selected = set(i for i, row in enumerate(rows) if row['phase'] in ('align', 'insert'))
    selected.add(max(i for i, row in enumerate(rows) if row['phase'] == 'transport'))
    measured = []
    with mujoco.Renderer(model, height=480, width=640) as renderer, mujoco.Renderer(model, height=480, width=640) as depth_renderer, (stats / 'privileged.jsonl').open() as source:
        renderer.enable_segmentation_rendering()
        depth_renderer.enable_depth_rendering()
        for index, line in enumerate(source):
            if index not in selected:
                continue
            extra = json.loads(line)
            restore(model, data, rows[index], extra, stats / 'initial_state.npz')
            plug = data.body('task/yellow_plug')
            rotation = plug.xmat.reshape(3, 3)
            tip_points = []
            for geom in groups['plug_pins']:
                mesh = model.geom_dataid[geom]
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
                world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                local_z = ((world - plug.xpos) @ rotation)[:, 2]
                tip_points.extend(world[np.isclose(local_z, local_z.min(), atol=1e-7)])
            tip_points = np.asarray(tip_points)
            for camera in ('overhead', 'left_wrist', 'right_wrist'):
                renderer.update_scene(data, camera=camera)
                segmentation = renderer.render()
                counts = {name: int(np.count_nonzero(np.isin(segmentation[:, :, 0], ids)
                          & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))) for name, ids in groups.items()}
                depth_renderer.update_scene(data, camera=camera)
                depth = depth_renderer.render()
                cid = model.camera(camera).id
                optical = (tip_points - data.cam_xpos[cid]) @ data.cam_xmat[cid].reshape(3, 3)
                optical *= np.array([1., -1., -1.])
                k = np.asarray(extra['cameras'][camera]['intrinsics'])
                pixels = optical @ k.T
                pixels = pixels[:, :2] / pixels[:, 2:3]
                gaps = []
                for point, pixel in zip(optical, pixels):
                    u, v = np.rint(pixel).astype(int)
                    if point[2] > 0 and 1 <= u < 639 and 1 <= v < 479:
                        # A 3x3 neighborhood tolerates rasterization at mesh corners.
                        gaps.append(float(point[2] - depth[v-1:v+2, u-1:u+2].max()))
                counts['tip_corners_visible_1mm'] = sum(abs(gap) <= .001 for gap in gaps)
                counts['min_tip_occlusion_gap_m'] = min(gaps) if gaps else None
                measured.append(dict(frame=index, phase=rows[index]['phase'], camera=camera, **counts))
    result = dict(episode=ident, role='PRIVILEGED_EVALUATOR_ONLY_NOT_STUDENT_INPUT',
                  method='Archived integration-state restoration and rendered visible geom-ID pixel counts; no physics steps',
                  scope='Native outlet 3; transport endpoint and every align/insert frame in this one Episode',
                  cameras=['overhead', 'left_wrist', 'right_wrist'], samples=len(measured),
                  pin_visible_samples=sum(row['plug_pins'] > 0 for row in measured),
                  pin_visible_frames=len({row['frame'] for row in measured if row['plug_pins'] > 0}),
                  tested_frames=len(selected),
                  per_camera={camera: dict(pin_visible_samples=sum(row['plug_pins'] > 0 for row in measured if row['camera'] == camera),
                              max_pin_pixels=max(row['plug_pins'] for row in measured if row['camera'] == camera),
                              frames_with_visible_tip_corners=sum(row['tip_corners_visible_1mm'] > 0 for row in measured if row['camera'] == camera))
                              for camera in ('overhead', 'left_wrist', 'right_wrist')},
                  limitation='Evaluator-only mesh/depth projection at pin end corners (1 mm, 3x3 raster tolerance); not Student input or semantic detector accuracy. Does not rule out a separately calibrated housing-to-tip estimator.',
                  evidence=[dict(dataset_relative_path=str(p.relative_to(DATA)), sha256=digest(p))
                            for p in (stats / 'model.xml', stats / 'privileged.jsonl', raw / 'trajectory.jsonl')])
    detail = DATA / 'self_evolution/event_residual_mvp' / ident / 'visibility_evaluator.json'
    save(detail, measured)
    result['detail'] = dict(dataset_relative_path=str(detail.relative_to(DATA)), sha256=digest(detail))
    save(ROOT / 'experiments/event_residual_mvp' / ('visibility_' + ident + '.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode_id')
    audit(parser.parse_args().episode_id)

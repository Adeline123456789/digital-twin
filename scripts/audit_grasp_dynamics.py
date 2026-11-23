"""Synchronize archived simulation aperture, hand-relative pose and finger contacts."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from learn import DATA, ROOT, digest, save


EPISODES = ('20261123T090549Z_a35ab76b', '20261123T092135Z_48f4ff46')


def classify_window(rows, visual_change_measured=False):
    command_range = np.ptp([r['commanded_aperture_m'] for r in rows]) * 1000
    actual_range = np.ptp([r['actual_aperture_m'] for r in rows]) * 1000
    positions = np.array([r['plug_in_gripper_position_m'] for r in rows])
    translation = float(np.max(np.linalg.norm(positions - positions[0], axis=1))) * 1000
    rotations = Rotation.from_matrix(np.array([r['plug_in_gripper_rotation'] for r in rows]))
    angle = float(np.max((rotations[0].inv() * rotations).magnitude())) * 180 / np.pi
    finger_presence = len({(r['left_finger_contact'], r['right_finger_contact']) for r in rows}) > 1
    topology = len({tuple(tuple(pair) for pair in r['finger_plug_geom_pairs']) for r in rows}) > 1 or finger_presence
    penetration = max(r['penetration_m'] for r in rows) * 1000
    if command_range > .05:
        label = 'CONTROL_COMMAND_CHANGE'
    elif penetration > .2:
        label = 'SIM_CONTACT_ARTIFACT_CANDIDATE'
    elif actual_range > .2 and (translation > .2 or angle > .5) and topology:
        label = 'POSSIBLE_SLIP_OR_RESEATING'
    elif actual_range <= .2 and translation <= .2 and angle <= .5 and visual_change_measured:
        label = 'VISUAL_PERSPECTIVE_ONLY'
    else:
        label = 'UNRESOLVED_DYNAMICS'
    return dict(classification=label, commanded_range_mm=float(command_range), actual_range_mm=float(actual_range),
                relative_translation_mm=translation, relative_rotation_deg=angle, contact_topology_changed=topology,
                finger_presence_changed=finger_presence,
                max_penetration_mm=penetration,
                privileged_event='IN_HAND_POSE_DRIFT' if translation > .2 or angle > .5 else None,
                event_role='PRIVILEGED_EVALUATION_ONLY', visual_change_measured=visual_change_measured)


def audit_episode(ident):
    import mujoco
    from contact_evidence import restore
    raw = DATA / 'self_evolution' / ident
    stats = DATA / 'complete_sim/succeed' / ident / 'proprio_stats'
    model = mujoco.MjModel.from_xml_path(str(stats / 'model.xml'))
    data = mujoco.MjData(model)
    history = [json.loads(line) for line in (raw / 'trajectory.jsonl').read_text().splitlines()]
    rows = []
    with (stats / 'privileged.jsonl').open() as stream:
        for index, (row, line) in enumerate(zip(history, stream, strict=True)):
            extra = json.loads(line)
            if extra['frame'] != index or extra['phase'] != row['phase']:
                raise ValueError('Unsynchronized physics/contact trace')
            restore(model, data, row, extra, stats / 'initial_state.npz')
            grip, plug = data.body('right/gripper_center'), data.body('task/yellow_plug')
            inverse = grip.xmat.reshape(3, 3).T
            pairs = [{c['body1'], c['body2']} for c in extra['contacts'] if c['distance_m'] <= 0]
            rows.append(dict(frame=index, time_s=row['time'], phase=row['phase'],
                             commanded_aperture_m=row['action'][13], actual_aperture_m=row['q'][13],
                             plug_in_gripper_position_m=(inverse @ (plug.xpos - grip.xpos)).tolist(),
                             plug_in_gripper_rotation=(inverse @ plug.xmat.reshape(3, 3)).tolist(),
                             left_finger_contact={'task/yellow_plug', 'right/link7'} in pairs,
                             right_finger_contact={'task/yellow_plug', 'right/link8'} in pairs,
                             finger_plug_geom_pairs=sorted({tuple(sorted((c['geom1'], c['geom2']))) for c in extra['contacts']
                                 if c['distance_m'] <= 0 and 'task/yellow_plug' in (c['body1'], c['body2'])
                                 and bool({'right/link7', 'right/link8'} & {c['body1'], c['body2']})}),
                             controller_target=row['action'], penetration_m=row['penetration']))
    differences = np.diff([r['actual_aperture_m'] for r in rows])
    center = int(np.argmin(differences)) + 1
    windows = {'fastest_narrowing': rows[max(0, center - 15):center + 16]}
    carry = []
    for index in range(len(rows) - 30):
        window = rows[index:index + 31]
        if (all(r['phase'] in ('lift', 'transport', 'align', 'insert') for r in window)
                and np.ptp([r['commanded_aperture_m'] for r in window]) <= .00005):
            carry.append(window)
    windows['constant_command_carry'] = max(carry, key=lambda w: np.ptp([r['actual_aperture_m'] for r in w]))
    destination = DATA / 'self_evolution/event_residual_mvp_v2/grasp_dynamics' / ident
    save(destination / 'synchronized_trace.json', rows)
    summaries = {}
    for name, window in windows.items():
        summaries[name] = dict(frame_start=window[0]['frame'], frame_end=window[-1]['frame'],
                               time_start_s=window[0]['time_s'], time_end_s=window[-1]['time_s'],
                               **classify_window(window))
        save(destination / (name + '.json'), window)
        plot_window(window, ROOT / 'experiments/event_residual_mvp_v2' / f'{ident}_{name}.svg', name)
    return dict(episode=ident, frames=len(rows), windows=summaries,
                trace=dict(dataset_relative_path=str((destination / 'synchronized_trace.json').relative_to(DATA)),
                           sha256=digest(destination / 'synchronized_trace.json')),
                evidence=[dict(dataset_relative_path=str(p.relative_to(DATA)), sha256=digest(p))
                          for p in (raw / 'trajectory.jsonl', stats / 'privileged.jsonl', stats / 'model.xml')])


def plot_window(rows, destination, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.array([r['time_s'] for r in rows])
    q = np.array([r['plug_in_gripper_position_m'] for r in rows]) * 1000
    rotation = Rotation.from_matrix(np.array([r['plug_in_gripper_rotation'] for r in rows]))
    angle = (rotation[0].inv() * rotation).magnitude() * 180 / np.pi
    fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(t, [r['commanded_aperture_m'] * 1000 for r in rows], '--', color='#c44e52', label='Command')
    axes[0].plot(t, [r['actual_aperture_m'] * 1000 for r in rows], color='#3569a8', label='Actual')
    axes[0].set_ylabel('Aperture (mm)'); axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(t, np.linalg.norm(q - q[0], axis=1), color='#3569a8', label='Translation (mm)')
    axes[1].plot(t, angle, color='#32875b', label='Rotation (deg)'); axes[1].legend(frameon=False, ncol=2)
    axes[1].set_ylabel('Hand-relative drift')
    axes[2].step(t, [int(r['left_finger_contact']) for r in rows], where='post', label='Finger link7')
    axes[2].step(t, [int(r['right_finger_contact']) + 1.2 for r in rows], where='post', label='Finger link8 + 1.2')
    axes[2].set_ylabel('Plug contact'); axes[2].legend(frameon=False, ncol=2)
    axes[3].plot(t, [r['penetration_m'] * 1000 for r in rows], color='#c44e52')
    axes[3].set_ylabel('Penetration (mm)'); axes[3].set_xlabel('Simulation time (s)')
    axes[0].set_title(f'{title}: {rows[0]["phase"]} → {rows[-1]["phase"]}')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata={'Date': None}); plt.close(fig)
    destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')


def main():
    results = [audit_episode(ident) for ident in EPISODES]
    save(ROOT / 'experiments/event_residual_mvp_v2/grasp_dynamics_audit.json',
         dict(scope='Archived simulation only; no new physics rollout and no parameter tuning',
              thresholds=dict(command_mm=.05, aperture_mm=.2, pose_mm=.2, rotation_deg=.5, penetration_mm=.2),
              episodes=results, classification_limit='30 Hz synchronized records; no real detector claim and no solver-causality intervention',
              real_force_observations=False, friction_changes=0))
    print(json.dumps([dict(episode=r['episode'], windows=r['windows']) for r in results], indent=2))


if __name__ == '__main__':
    main()

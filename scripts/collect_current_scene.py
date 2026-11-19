"""Run the existing complete collector with an explicit simulation embodiment."""
import argparse
import json
import os
from tempfile import TemporaryDirectory
import yaml
from pathlib import Path
from benchmark_collection import run_case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-arms', choices=['dual','right'], required=True)
    parser.add_argument('--offset-mm', type=float, default=0)
    parser.add_argument('--shift-mm', type=float, default=0)
    parser.add_argument('--start-right-deg', type=float, nargs=6)
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--relative-precontact', action='store_true',
                        help='Stop and verify at the pre-insertion gap in the current right-arm scene')
    parser.add_argument('--held-control', action='store_true', help='Start at archived simulated alignment; diagnostic only')
    parser.add_argument('--archived-regrasp', action='store_true', help='Held control from the separate 20 mm single-arm grasp')
    parser.add_argument('--socket-delta', nargs=3, type=float, metavar=('X_MM','Y_MM','YAW_DEG'))
    parser.add_argument('--stopped-rgb', type=Path, help='Stopped external RGB observation bundle for precontact prediction')
    args = parser.parse_args(argv)
    os.environ['PIPER_ACTIVE_ARMS'] = args.robot_arms
    os.environ['PIPER_SIM_PROFILE'] = 'original'
    case = dict(name=args.robot_arms, offset_mm=args.offset_mm, shift_mm=args.shift_mm,
                start_right_deg=args.start_right_deg)
    if args.relative_precontact:
        from task_relative import SCENE, HELD, REGRASP, DATA, read_stopped_rgb
        if args.robot_arms != 'right' or args.offset_mm or args.shift_mm:
            parser.error('Relative precontact uses the right arm and measured socket target')
        if args.archived_regrasp and not args.held_control:
            parser.error('--archived-regrasp requires --held-control')
        held = (REGRASP if args.archived_regrasp else HELD) if args.held_control else None
        source = DATA/'self_evolution'/held if held else SCENE.parent
        native = json.loads((source/'report.json').read_text())
        case.update(grasp_height_m=.022, align_gap_m=native['align_gap_m'],
                    start_right_deg=args.start_right_deg or native['initial_right_joint_deg'])
        options = dict(stage='precontact', held_control=args.held_control, held_episode=held,
                       socket_delta=args.socket_delta or [0.,0.,0.])
        if args.stopped_rgb:
            if args.socket_delta or args.held_control or args.start_right_deg:
                parser.error('Stopped RGB supplies its own scene and measured right-arm state')
            observation = read_stopped_rgb(args.stopped_rgb)
            if not observation.get('valid'):
                from task_relative import stop_measurement
                return stop_measurement(observation,args.stopped_rgb)
            options['measurement'] = observation
        case['relative_pose'] = options
        cfg = yaml.safe_load((source/'task.yaml').read_text())
        dx,dy,yaw = options['socket_delta']
        cfg['strip_xy_m'] = [v+delta/1000 for v,delta in zip(cfg['strip_xy_m'],(dx,dy))]
        cfg['strip_yaw_deg'] += yaw
        # The native collector sees only the scene and observations, never the test displacement.
        cfg['right_precontact'] = {k:v for k,v in options.items() if k != 'socket_delta'}
        cfg['right_precontact']['gap_m'] = case['align_gap_m']
        with TemporaryDirectory() as folder:
            scene = Path(folder)/'task.yaml'
            scene.write_text(yaml.safe_dump(cfg))
            if os.environ.get('PIPER_TARGET_INDEX','3') != '3':
                parser.error('The current right-only task requires outlet 3')
            old = os.environ.get('PIPER_TASK_CONFIG')
            os.environ['PIPER_TASK_CONFIG'] = str(scene)
            try:
                return run_case(case, True, label='Right-only precontact', kind='right_relative_pose', publish=False)
            finally:
                if old is None:
                    del os.environ['PIPER_TASK_CONFIG']
                else:
                    os.environ['PIPER_TASK_CONFIG'] = old
    elif args.held_control or args.archived_regrasp or args.socket_delta or args.stopped_rgb:
        parser.error('These inputs require --relative-precontact')
    return run_case(case, True, label='Right-arm adaptation; existing native task',
                    kind='right_arm_adaptation', publish=not args.local_only)


if __name__ == '__main__':
    main()

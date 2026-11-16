"""Run the existing complete collector with an explicit simulation embodiment."""
import argparse
import os
from benchmark_collection import run_case

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-arms',choices=['dual','right'],required=True)
    parser.add_argument('--offset-mm',type=float,default=0)
    parser.add_argument('--shift-mm',type=float,default=0)
    parser.add_argument('--local-only',action='store_true')
    args=parser.parse_args()
    os.environ['PIPER_ACTIVE_ARMS']=args.robot_arms
    os.environ['PIPER_SIM_PROFILE']='original'
    run_case({'name':args.robot_arms,'offset_mm':args.offset_mm,'shift_mm':args.shift_mm},True,
             label='Right-arm adaptation; existing native task',kind='right_arm_adaptation',publish=not args.local_only)

"""Compare a frozen native grasp-tilt candidate with old settings and preserve all Episodes."""
import json
import os
from learn import ROOT, save
from benchmark_collection import run_case


def main():
    folder=ROOT/'experiments/research_stage4/round2'
    assert os.environ.get('PIPER_SIM_PROFILE')=='retained_final'
    assert os.environ.get('PIPER_TARGET_INDEX')=='1'
    design=json.loads((folder/'design.json').read_text())
    selection=json.loads((folder/'selection.json').read_text())
    tilt=selection['selected_grasp_tilt_deg']
    if tilt is None:
        save(folder/'execution_progress.json',{'experiments':[],'status':'NO_FEASIBLE_SCREEN_CANDIDATE'})
        return
    experiments=[]
    for case in design['full_execution']['cases']:
        for arm in case['order']:
            condition=dict(name=case['role']+' '+arm,offset_mm=0,shift_mm=case['shift_mm'],
                grasp_tilt_deg=0 if arm=='baseline' else tilt,arm=arm,role=case['role'])
            result=run_case(condition,inline=False,label='Stage4 round2 native grasp repair',kind='research_stage4')
            experiments.append(result['experiment_id'])
            save(folder/'execution_progress.json',{'experiments':experiments,'status':'RUNNING'})
            if arm=='candidate' and not (result['task_success'] and result['recording_verified']):
                save(folder/'execution_progress.json',{'experiments':experiments,'status':'STOPPED_CANDIDATE_NOT_VALIDATED'})
                return
    save(folder/'execution_progress.json',{'experiments':experiments,'status':'COMPLETED'})


if __name__=='__main__':main()

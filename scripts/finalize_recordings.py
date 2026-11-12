"""Validate newly generated three-view recordings and place them under prescribed roots."""
import json
import argparse
from pathlib import Path
import shutil
import time
from learn import DATA,ROOT,record,finish,save,digest
from episode_format import supplement, validate
SUCCESS=DATA/'complete_sim/succeed/20261111_sim_calibrated_pick_insert_000001'
IDS=['20261111T083904Z_3457e2ad','20261111T083720Z_f64309a8']
def main():
    import av
    parser=argparse.ArgumentParser();parser.add_argument('ids',nargs='*');parser.add_argument('--stage',default='self_evolution/recording_stage');args=parser.parse_args()
    for ident in (args.ids or IDS):
        out,r=record('evaluation','Three-view recording verification '+ident,'simulation_recording')
        start=time.monotonic()
        try:
            stage=DATA/args.stage
            matches=list(stage.glob('complete_sim/*/learning_'+ident+'_000001'))
            if len(matches)!=1:raise RuntimeError('No completed recording archive for '+ident)
            source=matches[0];outcome=json.loads((source/'outcome.json').read_text())
            meta=json.loads((source/'proprio_stats/metadata.json').read_text())
            counts={}
            for f in (source/'videos').glob('*.mp4'):
                with av.open(str(f)) as video: counts[f.name]=sum(1 for _ in video.decode(video=0))
            if len(counts)!=3 or any(n!=meta['num_frames'] for n in counts.values()):raise ValueError('Video frame count mismatch')
            success=bool(outcome['success'])
            trial=DATA/'self_evolution'/ident
            supplement(source,trial)
            verification=validate(source,trial,expected_success=success)
            base=SUCCESS if success else DATA/'complete_sim/failed'/ident
            vd=base/'videos'/ident if success else base/'videos'
            md=base/'proprio_stats'/ident/'replay' if success else base/'proprio_stats'
            vd.parent.mkdir(parents=True,exist_ok=True);md.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(source/'videos',vd);shutil.copytree(source/'proprio_stats',md)
            save(md/'outcome.json' if success else base/'outcome.json',outcome)
            r.update(result='success',task_success=success,validation_status='SIMULATION_VERIFIED' if success else 'SIMULATION_FAILURE_RECORDED',
                initial_state={'source_experiment':ident},observed_effects=[{'outcome':outcome,'video_frames':counts,'quality':meta.get('quality')}],
                lessons_learned=['Videos, state/action and object trace verified using simulator privileged information; simulation-only evidence.'],
                next_improvement='Compare valid episode cost and sampled motion quality across matched simulation collection methods')
            r['evidence']=[{'dataset_relative_path':str(f.relative_to(DATA)),'sha256':digest(f)}
                           for folder in (vd,md) for f in folder.rglob('*') if f.is_file()]
            r['original_source_evidence']=str(source.relative_to(DATA))
            r['force_sensor_present']=False
            r['recording_verification']=verification
            r['privileged_information_used']=True
            r['privileged_fields_used']=['simulated joint/action state','object pose','collision penetration','task success ground truth']
            save(base/('recording_'+ident+'.json'),{'experiment_id':r['experiment_id'],'source':ident,'video_frames':counts,'success':success})
        except Exception as exc:r['error']=str(exc)
        finally:r['execution_time']=time.monotonic()-start;finish(out,r)
if __name__=='__main__':main()

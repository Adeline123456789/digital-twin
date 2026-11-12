"""Falsify a constant opening-coordinate offset before changing contact physics."""
import time
import numpy as np
from scipy.optimize import least_squares
from fit_replay_timing import PAIRS,load_pair
from learn import DATA,record,finish,digest


def plateau_mask(times,phases):
    end=times[phases=='grasp'][-1]
    return (phases=='grasp')&(times>=end-.5)


def main():
    out,r=record('research_stage2','Independent validation of constant aperture offset hypothesis','offline_historical')
    start=time.monotonic()
    try:
        pairs=[load_pair(*p) for p in PAIRS]
        t,y,st,sq,phase,_,_,_=pairs[0]
        mask=plateau_mask(t,phase)
        # Exploratory bound covers the earlier unvalidated 9.2 mm offset; not a hardware tolerance.
        fit=least_squares(lambda x:np.interp(t[mask],st,sq)+x[0]-y[mask],
                          [0.],bounds=([-12.],[12.]),max_nfev=20)
        offset=float(fit.x[0]);effects=[]
        for (name,_),role,pair in zip(PAIRS,('train','heldout'),pairs):
            t,y,st,sq,phase,_,valid,_=pair
            scopes=[('grasp_plateau',plateau_mask(t,phase))]+[(p,valid&(phase==p)) for p in ('approach','lift','transport','align','insert')]
            results=[]
            for scope,mask in scopes:
                baseline=np.interp(t[mask],st,sq)-y[mask]
                results.append({'phase':scope,'samples':int(mask.sum()),
                    'baseline_rmse_mm':float(np.sqrt(np.mean(baseline**2))),
                    'candidate_rmse_mm':float(np.sqrt(np.mean((baseline+offset)**2))),
                    'real_mean_width_mm':float(y[mask].mean()),
                    'sim_mean_width_mm':float(np.interp(t[mask],st,sq).mean())})
            effects.append({'source_episode':name,'role':role,'phases':results})
        r.update(result='success',task_success=None,validation_status='HYPOTHESIS_TESTED',
            initial_state={'train_episode':PAIRS[0][0],'heldout_episode':PAIRS[1][0],
                           'fit_window':'last 0.5 s of training grasp; frozen before fit'},
            actions=[{'optimizer':'scipy.optimize.least_squares','parameter':'constant_predicted_width_offset_mm','bounds':[-12,12],'max_nfev':20}],
            expected_effects=['A physical coordinate offset should transfer across phases, not only fit one contact plateau'],
            observed_effects=effects,
            repair={'candidate_offset_mm':offset,'nfev':fit.nfev,'model_mutation':False,
                    'accepted':all(x['candidate_rmse_mm']<=x['baseline_rmse_mm'] for x in effects[1]['phases'])},
            root_cause='UNKNOWN: contact geometry, object orientation and motor-to-aperture mapping are not separately identified',
            root_cause_status='unknown',
            lessons_learned=['Test the same aperture correction in precontact and held-object phases before changing geometry.'],
            next_improvement='Use existing independently observed fingertip/object geometry to distinguish contact and mapping hypotheses',
            evidence=[{'dataset_relative_path':str(p.relative_to(DATA)),'sha256':digest(p)} for pair in pairs for p in pair[-1]])
        r['limitations']=['Tests constant offset only, not every nonlinear or load-dependent mapping',
            'No new physical simulation or real motion; unchanged historical simulation model',
            'A shared task plan limits independence to recording/initial grasp differences',
            'Encoder change alone does not prove real slip, rotation or deformation']
    except Exception as exc:r['error']=str(exc)
    finally:
        r['analysis_version']=digest(__file__);r['tokens']={'planner':0,'research_orchestrator':None}
        r['execution_time']=time.monotonic()-start;r['simulation_seconds']=0;finish(out,r)


if __name__=='__main__':main()

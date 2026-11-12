"""Bounded observation-delay identification on existing replay outputs, not physics calibration."""
import csv
import json
import time
import numpy as np
from scipy.optimize import least_squares
from learn import DATA,record,finish,digest

PAIRS=(('20261109_real_residual_attempt_000001','20261111T092441Z_9c96874a'),
       ('20261109_real_grip_hold_attempt_000001','20261111T092442Z_599aa303'))
MAX_DELAY=.3


def fit_delay(times, observed, sim_times, predicted):
    return least_squares(lambda x:np.interp(times-x[0],sim_times,predicted)-observed,
                         [.05],bounds=([0.],[MAX_DELAY]),max_nfev=30)


def load_pair(name,ident):
    csvfile=DATA/'complete_real/failed'/name/'raw/task/commanded_probe.csv'
    trajectory=DATA/'self_evolution'/ident/'trajectory.jsonl'
    real=list(csv.DictReader(csvfile.open()))
    sim=[json.loads(x) for x in trajectory.read_text().splitlines()]
    t=np.array([float(x['monotonic']) for x in real]);t-=t[0]
    st=np.array([x['time'] for x in sim]);sq=np.array([x['q'][13]*1000 for x in sim])
    y=np.array([float(x['feedback_gripper_open'])*95 for x in real])
    u=np.array([float(x['target_gripper_open'])*95 for x in real])
    phase=np.array([x['phase'] for x in real])
    # Exclude interpolation extrapolation for every candidate, with a fixed window.
    valid=(t>=st[0]+MAX_DELAY)&(t<=st[-1])
    fit=valid&(phase=='grasp')&(u>=55)&(u<=59.5)
    return t,y,st,sq,phase,fit,valid,[csvfile,trajectory]


def evaluate(pair,delay):
    t,y,st,sq,phase,fit,valid,_=pair
    result=[]
    for label,mask in [('precontact_fit_window',fit)]+[(p,valid&(phase==p)) for p in dict.fromkeys(phase)]:
        old=np.interp(t[mask],st,sq)-y[mask]
        new=np.interp(t[mask]-delay,st,sq)-y[mask]
        result.append({'phase':str(label),'samples':int(mask.sum()),
            'baseline_rmse_mm':float(np.sqrt(np.mean(old**2))),
            'candidate_rmse_mm':float(np.sqrt(np.mean(new**2)))})
    return result


def main():
    out,r=record('research_stage2','Bounded replay observation-delay identification; held-out episode','offline_historical')
    start=time.monotonic()
    try:
        train,test=[load_pair(*pair) for pair in PAIRS]
        t,y,st,sq,_,mask,_,_=train
        fit=fit_delay(t[mask],y[mask],st,sq);delay=float(fit.x[0])
        evaluations=[{'source_episode':name,'role':role,'phases':evaluate(pair,delay)}
                     for (name,_),role,pair in zip(PAIRS,('train','heldout'),(train,test))]
        r.update(result='success',task_success=None,validation_status='MEASUREMENT_MODEL_CANDIDATE_ONLY',
            initial_state={'train_episode':PAIRS[0][0],'heldout_episode':PAIRS[1][0],
                           'fit_window':'grasp command between 55 and 59.5 mm; fixed before optimization'},
            actions=[{'optimizer':'scipy.optimize.least_squares','parameter':'predicted_observation_delay_s',
                      'bounds':[0,MAX_DELAY],'max_nfev':30}],
            expected_effects=['A timing-only correction generalizes to precontact motion; contact plateau remains unexplained'],
            observed_effects=evaluations,
            repair={'candidate_delay_s':delay,'nfev':fit.nfev,'optimizer_success':bool(fit.success),
                    'model_mutation':False,'accepted_for_physics':False},
            root_cause='UNKNOWN: fitted effective lag combines actuator response, replay discretization and observation timing',
            root_cause_status='unknown',
            lessons_learned=['Measurement alignment is not a physical contact-parameter calibration.'],
            next_improvement='Resolve aperture-coordinate versus grasp geometry identifiability using independent physical observations',
            evidence=[{'dataset_relative_path':str(p.relative_to(DATA)),'sha256':digest(p)}
                      for pair in (train,test) for p in pair[-1]])
        r['limitations']=['Only one precontact motion fitted; holdout shares the planned task sequence',
            'A bounded effective lag cannot identify hardware transport delay separately from plant lag',
            'No contacts, masses, friction, XML, task commands, or controller timing changed',
            'No independent real aperture measurement; encoder widths are not calibrated fingertip geometry']
    except Exception as exc:r['error']=str(exc)
    finally:
        r['analysis_version']=digest(__file__)
        r['tokens']={'planner':0,'research_orchestrator':None}
        r['execution_time']=time.monotonic()-start
        r['simulation_seconds']=0
        finish(out,r)


if __name__=='__main__':main()

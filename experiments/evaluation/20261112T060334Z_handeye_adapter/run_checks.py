"""Software verification only: analytic RGB projections, no real calibration fit."""
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
from test_rgb_handeye import synthetic_observations,run_adapter
from learn import save,digest,version
out=Path(__file__).parent;started=time.monotonic()
original=json.loads((ROOT/'experiments/research_stage2/20261112T052851Z_68d83a86/result.json').read_text())['program_version']['source_sha256']
assert original==version()['source_sha256']
data,truth=synthetic_observations()
save(out/'synthetic_input.json',data)
code,normal=run_adapter(data)
assert code==0
np.testing.assert_allclose(normal['T_gripper_camera'],truth,atol=1e-6)
cases=[dict(case='nominal_metric_RGB_PnP',returncode=code,result=normal)]
for i in range(6,10):data['T_camera_target'][i][0][3]+=.02
code,shift=run_adapter(data)
assert code==0
np.testing.assert_array_equal(shift['T_gripper_camera'],normal['T_gripper_camera'])
assert shift['candidate']['heldout']['translation_rmse_m']>.019
cases.append(dict(case='heldout_target_shift_fit_must_not_change',returncode=code,result=shift))
for name in ['single_axis','wrong_units','outside_bounds']:
 data,_=synthetic_observations(single_axis=name=='single_axis')
 if name=='wrong_units':data['translation_unit']='mm'
 if name=='outside_bounds':data['bounds']['max_translation_change_m']=.001
 code,result=run_adapter(data)
 assert code==2
 cases.append(dict(case=name,returncode=code,result=result))
save(out/'case_results.json',cases)
prep=json.loads((ROOT/'.local/rgb_handeye_adapter/multiaxis_robot_poses.json').read_text())
summary=dict(experiment_id=out.name,timestamp=datetime.now(timezone.utc).isoformat(),model='OpenCV Park-Martin solver; no planner LLM',
 task='Verify mature hand-eye solver adapter and existing FK input connection',environment='synthetic_software_verification_and_archive_input_preparation',
 program_version={f:digest(ROOT/f) for f in ['scripts/solve_rgb_handeye.py','scripts/prepare_rgb_handeye.py','tests/test_rgb_handeye.py']},
 protocol_version=digest(ROOT/'protocol/robot_problem_solving.md'),memory_version='No physical skill or calibration promoted',
 initial_state={'synthetic_frames':10,'train_frames':6,'heldout_frames':4,'real_capture_selected_frames':prep['frame_indices']},
 actions=['Run RGB projection/PnP fixture through installed system OpenCV hand-eye API','Verify holdout isolation and reject invalid/degenerate inputs','Use original Piper URDF FK on existing measured feedback'],
 expected_effects=['Recover synthetic known transform without using heldout poses or depth','Prepare real robot-side poses without claiming visual pose availability'],
 observed_effects={'nominal_baseline':normal['baseline'],'nominal_candidate':normal['candidate'],
 'nominal_max_matrix_element_error':float(np.max(abs(np.array(normal['T_gripper_camera'])-truth))),
 'heldout_shift_error_m':shift['candidate']['heldout']['translation_rmse_m'],'expected_rejections':3,
 'real_input_status':prep['status'],'real_capture_phases':prep['phase']},
 result='success',task_success=None,root_cause='UNKNOWN: no new physical failure diagnosis or real calibration fit',root_cause_status='unknown',
 validation_status='SOFTWARE_VERIFIED_ONLY',repair={'calibration_promoted':False,'physical_parameters_changed':False,'new_dependencies':False},
 real_input_provenance_hashes=prep['provenance']['input_sha256'],fk_source_sha256=prep['provenance']['fk_source_sha256'],
 original_simulation_sources_unchanged_since_round6=True,new_research_round=False,stage2_rounds_used=6,
 tokens={'planner':0,'research_orchestrator':None},execution_time=time.monotonic()-started,simulation_seconds=0,robot_execution_seconds=0,human_interventions=None,
 lessons_learned=['Installed Python environments expose different OpenCV APIs; reuse the available solver without changing the simulation environment','Synthetic hand-eye correctness and closure residuals do not establish real calibration'],
 next_improvement='Obtain qualified RGB target/camera poses for existing multi-axis capture; handle planar ambiguity and metric-scale provenance before any real calibration fit')
save(out/'result.json',summary)
print(json.dumps(summary['observed_effects'],indent=2))

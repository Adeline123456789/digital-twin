from pathlib import Path
import sys,json,numpy as np,yaml,cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from itertools import permutations
sys.path.insert(0,str(Path.cwd()/'scripts'))
from learn import DATA,save,digest
from plan_board_views import MujocoBackend,load,body_pose
B=DATA/'self_evolution/end_to_end_cycle10';C=B/'current_comparison_001';R=DATA/'self_evolution/current_scene_cycle11';O=R/'fixture_fit_001';O.mkdir(exist_ok=False)
views=json.loads((C/'paired_views.json').read_text())['views'];mount=np.array(json.loads((R/'orientation_fit_001/private_mount.json').read_text())['T_parent_rgb']);K=np.array(json.loads((B/'episodes/20261116_real_current_location_000002/raw/task/right_wrist/camera.json').read_text())['K']);task=yaml.safe_load((B/'current_location_probe_002/task.yaml').read_text());s=MujocoBackend(dict(load(),task_scene=task));rows=[];private={}
def corner_pixels(phase,region):
 im=cv2.imread(str(C/(phase+'_right_wrist.jpg')));g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);h=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);yy,xx=np.indices(g.shape)
 mask=((g<35)&(xx>260)&(xx<470)&(yy<240)).astype('uint8') if region=='holder' else cv2.inRange(h,np.array([90,50,30]),np.array([125,255,255]))
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 if region=='holder':contour=max(contours,key=cv2.contourArea)
 else:
  good=[c for c in contours if cv2.contourArea(c)>6000 and cv2.boundingRect(c)[1]>5 and sum(cv2.boundingRect(c)[1::2])<400];good.sort(key=lambda c:cv2.moments(c)['m01']/cv2.moments(c)['m00']);contour=good[-1] if 'alternate' in phase else good[0]
 approx=cv2.approxPolyDP(contour,.03*cv2.arcLength(contour,True),True).reshape(-1,2)
 if len(approx)!=4:approx=cv2.boxPoints(cv2.minAreaRect(contour))
 return approx.astype(float)
try:
 for region,body,indices in [('holder','task/board',[0,1]),('socket','task/strip',[2,3])]:
  pose=body_pose(s,body)
  if region=='holder':xy=np.array(s.task.fixture.assets['holes'][5]['polygon_xy_m']);z=.033
  else:
   x,y=s.task.outlets[2]+[0,.006];xy=np.array([[x-.018,y-.0185],[x-.018,y+.0185],[x+.018,y+.0185],[x+.018,y-.0185]]);z=s.task.socket_face_z
  local=np.c_[xy,np.full(4,z)];cameras=[];observed=[]
  def points(delta):return local@(Rotation.from_euler('z',delta[2]).as_matrix()@pose[:3,:3]).T+pose[:3,3]+[*delta[:2],0]
  def project(delta,i):
   cam=cameras[i];p=(points(delta)-cam[:3,3])@cam[:3,:3];uv=p@K.T;return uv[:,:2]/uv[:,2:]
  for j,index in enumerate(indices):
   v=views[index];q=np.array(v['images']['right_wrist']['q_deg_width_m']);q[:6]=np.deg2rad(q[:6]);s.set_observation_state(q);cameras.append(body_pose(s,'right/link6')@mount);obs=corner_pixels(v['phase'],region);pred=project([0,0,0],j);centered=pred-pred.mean(0);perm=min(permutations(range(4)),key=lambda order:np.linalg.norm((obs[list(order)]-obs.mean(0))-centered));observed.append(obs[list(perm)])
  fit=least_squares(lambda x:(project(x,0)-observed[0]).ravel(),[0,0,0],bounds=([-.05,-.05,-np.deg2rad(15)],[.05,.05,np.deg2rad(15)]),max_nfev=100)
  before=[float(np.sqrt(np.mean(np.sum((project([0,0,0],i)-observed[i])**2,axis=1)))) for i in range(2)];after=[float(np.sqrt(np.mean(np.sum((project(fit.x,i)-observed[i])**2,axis=1)))) for i in range(2)]
  passed=bool(fit.success and not fit.active_mask.any() and after[1]<=3 and after[1]<.5*before[1]);rows.append(dict(region=region,delta_xy_mm=(fit.x[:2]*1000).tolist(),delta_yaw_deg=float(np.rad2deg(fit.x[2])),baseline_corner_rmse_px=before,candidate_corner_rmse_px=after,heldout_pixel_gate_passed=passed,active_bounds=fit.active_mask.tolist(),train_phase=views[indices[0]]['phase'],heldout_phase=views[indices[1]]['phase']))
  private[region]=dict(observed=[x.tolist() for x in observed],predicted=[project(fit.x,i).tolist() for i in range(2)],delta=fit.x.tolist())
 save(O/'private_fit.json',private);save(O/'result.json',dict(results=rows,scope='Known native fixture geometry, fixed table height and current RGB; one view fit, one view check. No physical calibration claim or deployment.',source_episode='20261116_real_current_location_000002',camera_candidate_hash=digest(R/'orientation_fit_001/private_mount.json'),private_fit_sha256=digest(O/'private_fit.json'),hardware_commands=0));print(json.dumps(rows))
finally:s.close()

"""Build and verify standard three-camera connector_mixed.v1 episodes before publication."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import av
import h5py
import numpy as np
import yaml
from learn import DATA, ROOT, digest, save

FAILED=DATA/'complete_sim/failed'
REFERENCE=FAILED/'20261111_sim_real05_failure_replay_000001'
ORIGINAL=Path('/media/midea/F496AC8096AC4548/piper_simulation')
RETAINED=DATA/'calibration/offline_workspace_20261110'
CAMERAS=('cam_external_rgb.mp4','cam_left_wrist_rgb.mp4','cam_right_wrist_rgb.mp4')

def source_for(record):
    hashes=record['program_version']['source_sha256']
    source=RETAINED if 'configs/tasks/real_connector_final.yaml' in hashes else ORIGINAL
    for rel,expected in hashes.items():
        if rel.startswith(('src/','scripts/','configs/')) and digest(source/rel)!=expected:
            raise ValueError('Source changed since trial: '+rel)
    return source

def fingerprint(trial,source):
    cfg=yaml.safe_load((trial/'task.yaml').read_text());camera=Path(cfg['camera_calibration'])
    if not camera.is_absolute():camera=source/camera
    cfg['camera_calibration_sha256']=digest(camera);del cfg['camera_calibration']
    h=hashlib.sha256((str(source)+json.dumps(cfg,sort_keys=True)+digest(trial/'trajectory.jsonl')).encode())
    with np.load(trial/'initial_state.npz') as state:
        for key in sorted(state.files):h.update(key.encode()+state[key].tobytes())
    return h.hexdigest()

def validate(episode,trial,expected_success=False):
    meta=json.loads((episode/'proprio_stats/metadata.json').read_text())
    ref=json.loads((REFERENCE/'proprio_stats/metadata.json').read_text())
    assert meta['schema']==ref['schema'] and meta['units']==ref['units']
    assert meta['domain']=='sim' and meta['success'] is expected_success and meta['force_sensor_present'] is False
    rows=[json.loads(x) for x in (trial/'trajectory.jsonl').read_text().splitlines()]
    n=meta['num_frames'];assert n==len(rows) and n>0, 'Incomplete replay'
    with h5py.File(REFERENCE/'proprio_stats/proprio_stats.h5') as f:
        fields={};f.visititems(lambda k,v:fields.update({k:(v.shape[1:],str(v.dtype))}) if isinstance(v,h5py.Dataset) else None)
    with h5py.File(episode/'proprio_stats/proprio_stats.h5') as f:
        for key,(shape,dtype) in fields.items():assert f[key].shape==(n,*shape) and str(f[key].dtype)==dtype,(key,'schema mismatch')
        q=np.array([x['q'] for x in rows]);actions=np.array([x['action'] for x in rows])
        for arm,offset in [('left_arm',0),('right_arm',7)]:
            assert np.max(abs(np.deg2rad(f[f'state/{arm}/joint_positions'][:])-q[:,offset:offset+6]))<=1e-5
            assert np.max(abs(f[f'state/{arm}/gripper_width_m'][:]-q[:,offset+6]))<=1e-5
            assert np.max(abs(np.deg2rad(f[f'action/{arm}/joint_positions'][:])-actions[:,offset:offset+6]))<=1e-8
        assert np.max(abs(f['sim_time'][:]-np.array([x['time'] for x in rows])))<=1e-8
    frames={}
    for camera in CAMERAS:
        with av.open(str(episode/'videos'/camera)) as video:frames[camera]=sum(1 for _ in video.decode(video=0))
        assert frames[camera]==n,'Video frame count mismatch'
    privileged=[json.loads(x) for x in (episode/'proprio_stats/privileged.jsonl').read_text().splitlines()]
    assert len(privileged)==n
    object_error=max(np.max(abs(np.array(row['objects']['yellow_plug']['position'])-source['plug'])) for row,source in zip(privileged,rows))
    assert object_error<=1e-6, 'Object trajectory mismatch'
    for name in ('model.xml','initial_state.npz','recorded_commands.csv'):assert (episode/'proprio_stats'/name).is_file()
    return dict(video_frames=frames,quality=meta['quality'],duration_s=n/meta['fps'],
        object_replay_max_error_m=float(object_error),success=expected_success,
        status='Format and replay verified; outcome preserved from the independent collector')

def supplement(episode,trial):
    stats=episode/'proprio_stats'
    shutil.copy2(trial/'initial_state.npz',stats/'initial_state.npz')
    meta=json.loads((stats/'metadata.json').read_text())
    # Explicit simulated commands/feedback in degrees and normalized gripper units; no fake real timestamps.
    with h5py.File(stats/'proprio_stats.h5') as f,(stats/'recorded_commands.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['monotonic','phase',*[f'target_j{i}' for i in range(1,7)],*[f'feedback_j{i}' for i in range(1,7)],'target_gripper_open','feedback_gripper_open'])
        for i in range(meta['num_frames']):
            writer.writerow([f['timestamp_monotonic'][i],meta['quality']['phase_names'][int(f['phase_index'][i])],*f['action/right_arm/joint_positions'][i],*f['state/right_arm/joint_positions'][i],f['action/right_arm/gripper_open'][i],f['state/right_arm/gripper_open'][i]])
    meta['recorded_commands_provenance']='Replayed simulation target/state samples, NOT real CAN feedback'
    durable=DATA/'self_evolution'/(trial.parent.name if trial.name=='source_trial' else trial.name)
    if not durable.is_dir() or digest(durable/'trajectory.jsonl')!=digest(trial/'trajectory.jsonl'):
        raise ValueError('No durable matching raw trial outside final dataset')
    meta['source_trial']=str(durable.relative_to(DATA))
    save(stats/'metadata.json',meta)
    with h5py.File(stats/'proprio_stats.h5','r+') as f:
        f.attrs['recorded_commands_provenance']=meta['recorded_commands_provenance'];f.attrs['source_trial']=meta['source_trial']

def convert_group(group,work):
    first=group[0];ident=first['id'];trial=Path(first.get('trial',str(FAILED/ident/'source_trial')));stage=work/ident;stage.mkdir()
    source=Path(first['source']);runtime=source/('.venv-mujoco/bin/python' if source==RETAINED else '.venv-mujoco-hardware/bin/python')
    if (FAILED/ident/'videos').is_dir():
        episode=stage/'ready';episode.mkdir()
        for name in ('videos','proprio_stats'):shutil.copytree(FAILED/ident/name,episode/name)
        meta=json.loads((episode/'proprio_stats/metadata.json').read_text())
        save(episode/'outcome.json',dict(domain='sim',success=False,source_session=meta['session_id'],source_episode=meta['episode_id'],policy_training_eligible=False))
    else:
        cmd=[str(runtime),str(source/'scripts/collect_plug_sim_episode.py'),'--trial',str(trial),'--root',str(stage),'--session','format_'+ident]
        with (stage/'conversion.log').open('w') as log:
            result=subprocess.run(cmd,cwd=source,env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if result.returncode:raise RuntimeError('Collector failed: '+(stage/'conversion.log').read_text()[-1200:])
        episode=stage/'complete_sim/failed'/('format_'+ident+'_000001')
    supplement(episode,trial);verification=validate(episode,trial)
    results=[]
    for item in group:
        dest=FAILED/item['id'];incoming=stage/('install_'+item['id']);shutil.copytree(episode,incoming)
        # A shared render is reused only for identical config, initial state, cameras and full trajectory.
        item_trial=Path(item.get('trial',str(dest/'source_trial')))
        supplement(incoming,item_trial);verified=validate(incoming,item_trial)
        verified['conversion_source_trial_sha256']=item['trajectory_sha256'];verified['equivalent_render_source']=ident
        save(incoming/'verification.json',verified)
        backup=stage/('old_'+item['id'])
        if dest.exists():dest.rename(backup)
        incoming.rename(dest)
        if backup.exists():shutil.rmtree(backup)
        results.append(dict(id=item['id'],status='converted',verification=verified))
        print('CONVERTED',item['id'],flush=True)
    # Remove only this job's disposable duplicate encoding outputs after successful installation.
    shutil.rmtree(stage)
    return results

"""Read completed simulation episodes. All timestamps for simulated motion use sim_time."""
import json
import hashlib
from dataclasses import replace
from pathlib import Path
from .contracts import Observation, Reference
from .first_divergence import event, report
from .legacy import DATA, ROOT, digest
from .timeline import aligned_samples
from diagnose_retention import analyze_retention


def reference(path, root, repository=False):
    return Reference('repository' if repository or root == ROOT else 'dataset', str(path.relative_to(root)), digest(path))


def json_file(path):
    return json.loads(path.read_text())


class EventBuilder:
    def __init__(self, data=DATA, repository=ROOT):
        self.data = data
        self.repository = repository

    def stopped_local(self, path, nominal_manifest, policy_hash):
        """Terminal local experiments are diagnostic evidence, not formal Episodes."""
        path = Path(path).resolve()
        if not path.is_relative_to(self.data / 'self_evolution/event_residual_mvp_v2/evaluation'):
            raise ValueError('Expected a stopped V2 local evaluation trace')
        result = json_file(path)
        if not result['trace'] or not isinstance(result['is_success'], bool):
            raise ValueError('Missing terminal outcome or trace')
        refs = (reference(path, self.data),)
        last = result['trace'][-1]
        ident = result['case_id'] + '_' + result['variant']
        fact = Observation(ident, 'original', nominal_manifest['source_hash'], nominal_manifest['task_config_hash'],
            policy_hash, 'plug_insertion', 3, last['phase'], str(last['frame']), (last['frame'] + 1) / 30,
            'sim_time', 'verified', 'simulator', 'status', 'sim.native_report', 'observed',
            dict(success=result['is_success'], collision=result['collision']), refs)
        if fact.issue():
            raise ValueError(fact.issue())
        events = [] if result['is_success'] else [event(fact, 'NATIVE_TASK_FAILED',
            status='TERMINAL_AFTER_LAST_SAMPLE', severity='danger' if result['collision'] else 'warning')]
        return report(ident, events, result['is_success'], refs)

    def build(self, path, observer=None):
        path = Path(path).resolve()
        r = json_file(path)
        ident = r['experiment_id']
        refs = (reference(path, self.repository, repository=True),)
        success = r.get('task_success')
        action = next(iter(r.get('actions') or []), {})
        adapter=r.get('program_version',{}).get('adapter_sha256')
        policy=hashlib.sha256(json.dumps([adapter,action],sort_keys=True).encode()).hexdigest() if adapter else None
        fact = Observation(ident, action.get('profile','unknown') if isinstance(action,dict) else 'unknown',
                           r.get('program_version',{}).get('source_sha256',{}).get('scripts/try_plug_pick_sim.py',''),
                           r.get('initial_state',{}).get('task_config_sha256'), policy,
                           'plug_insertion',r.get('initial_state',{}).get('target_index'), 'unknown','unknown',None,
                           'sim_time','unverified','world','m','sim.privileged_contact','unavailable',None,refs)
        def unknown(kind):
            return report(ident,[event(fact,kind)],success,refs)
        if r.get('result')=='infrastructure_error':
            return unknown('INFRA_ERROR')
        if path.parent.name != ident or not isinstance(success,bool) or not r.get('recording_verified'):
            return unknown('SENSOR_UNAVAILABLE')
        try:
            ep = self.data/'complete_sim'/('succeed' if success else 'failed')/ident
            stats = ep/'proprio_stats'
            trace = self.data/'self_evolution'/ident/'trajectory.jsonl'
            priv = stats/'privileged.jsonl'
            meta = json_file(stats/'metadata.json')
            verified = json_file(ep/f'recording_{ident}.json')
            outcome = json_file(ep/'outcome.json')
            if (meta['domain']!='sim' or outcome['domain']!='sim' or outcome['success'] is not success or meta['success'] is not success
                    or meta['source_trial'] != str(trace.parent.relative_to(self.data)) or verified['source'] != ident):
                return unknown('VERSION_UNKNOWN')
            if meta['fps']<=0 or meta['units'].get('gripper_width_m')!='metre' or meta['units'].get('end_effector_pos')!='mm':
                return unknown('FRAME_OR_UNITS_MISMATCH')
            names = {'cam_external_rgb.mp4','cam_right_wrist_rgb.mp4'}
            if r.get('active_arms') != ['right']:
                names.add('cam_left_wrist_rgb.mp4')
            if any(verified['video_frames'].get(name)!=meta['num_frames'] or not (ep/'videos'/name).stat().st_size for name in names):
                return unknown('SENSOR_UNAVAILABLE')
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            extras = [json.loads(line) for line in priv.read_text().splitlines()]
            import h5py
            with h5py.File(stats/'proprio_stats.h5') as h5:
                if len(rows)!=meta['num_frames'] or not aligned_samples(rows, extras, h5['sim_time'][:],meta['fps']):
                    return unknown('TIMESTAMP_MISMATCH')
            refs += tuple(reference(p,self.data) for p in (trace,priv,ep/f'recording_{ident}.json',stats/'metadata.json',stats/'proprio_stats.h5',ep/'outcome.json'))
            validation_path = (self.repository/r['recording_report']/'result.json').resolve()
            if not validation_path.is_relative_to(self.repository/'experiments'):
                return unknown('VERSION_UNKNOWN')
            validation = json_file(validation_path)
            if (verified['experiment_id']!=validation['experiment_id'] or validation['result']!='success'
                    or validation['initial_state'].get('source_experiment')!=ident or validation['task_success'] is not success):
                return unknown('VERSION_UNKNOWN')
            def pins(record):
                return {Path(item['dataset_relative_path']).name:item['sha256'] for item in record['evidence']}
            recording_pins,trial_pins=pins(validation),pins(r)
            if trial_pins.get(trace.name)!=refs[1].sha256 or any(recording_pins.get(Path(ref.path).name)!=ref.sha256
                    for ref in refs if Path(ref.path).name in {priv.name,'proprio_stats.h5','metadata.json'}):
                return unknown('VERSION_UNKNOWN')
            if any(digest(ep/'videos'/name)!=recording_pins.get(name) for name in names):
                return unknown('SENSOR_UNAVAILABLE')
            refs += (reference(validation_path,self.repository,repository=True),)
            fact = replace(fact, evidence_refs=refs, time_alignment_quality='verified', observability='observed')
            if observer:
                for i,(row,extra) in enumerate(zip(rows,extras)):
                    observer(meta['fps'],replace(fact,phase=row['phase'],action_id=str(i),timestamp_mono=row['time'],
                                     observed_value=dict(plug=row['plug'],grip=row['grip'],contacts=extra['contacts'])))
            result = from_trace(fact,rows,extras,r['observed_effects'][0],success)
            if not success and result.status in {'NO_DIVERGENCE_OBSERVED','NATIVE_TASK_FAILED'} and self.data == DATA:
                from agent_challenge import precontact_observation
                if trial_pins.get('task.yaml')!=digest(trace.parent/'task.yaml'):
                    return unknown('VERSION_UNKNOWN')
                residual = precontact_observation(ident).get('observed_precontact_xy_error_mm')
                if residual and max(map(abs,residual)) >= 1.0:
                    i = max(j for j,row in enumerate(rows) if row['phase']=='align')
                    refs += (reference(trace.parent/'task.yaml', self.data),)
                    o = replace(fact,phase='align',action_id=str(i),timestamp_mono=rows[i]['time'],
                                frame_id='socket',units='mm',sensor_provenance='sim.socket_residual',
                                observed_value=residual,predicted_value=[0.,0.],residual=residual,evidence_refs=refs)
                    return report(ident,[event(o,'TARGET_FRAME_RESIDUAL',first_observed_frame=i,
                                  first_observed_time=o.timestamp_mono,confirmation_time=o.timestamp_mono,status='OBSERVED'),*result.events],success,refs)
            return result
        except (OSError, KeyError, ValueError, TypeError, IndexError):
            return unknown('SENSOR_UNAVAILABLE')


def from_trace(fact,rows,extras,native,success):
    d = analyze_retention(rows,extras)
    events = []
    if d['status']=='GRASP_RETENTION_LOSS_OBSERVED':
        i = d['earliest_observed_frame']
        o = replace(fact,phase=rows[i]['phase'],action_id=str(i),timestamp_mono=rows[i]['time'],
                    observed_value=d['first_event_distance_mm']/1000,
                    predicted_value=d['lift_reference_distance_mm']/1000)
        events.append(event(o,d['status'],first_observed_time=o.timestamp_mono,first_observed_frame=i,
                            confirmation_time=d['confirmation_time_s'],status='OBSERVED'))
    elif d['status']=='UNKNOWN':
        events.append(event(fact,'SENSOR_UNAVAILABLE'))
    last = rows[-1]
    o = replace(fact,phase=native.get('last_phase',last['phase']),action_id='native_terminal',
                timestamp_mono=last['time'],sensor_provenance='sim.native_report',frame_id='simulator',units='status',
                observed_value=native.get('error') or 'complete')
    if o.issue():
        events.append(event(o,o.issue()))
    elif native.get('error'):
        kind = 'IK_UNREACHABLE' if 'unreachable' in str(native['error']).lower() else 'NATIVE_TASK_FAILED'
        events.append(event(o,kind,status='TERMINAL_AFTER_LAST_SAMPLE'))
    return report(fact.experiment_id,events,success,fact.evidence_refs)

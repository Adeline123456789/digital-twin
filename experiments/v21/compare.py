"""Offline paired benchmark. Run with the project's simulator Python; stdout is JSON."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT)]


def read_json(path):
    return json.loads(path.read_bytes())


def shared_feedback(case, data):
    """Read the same archived feedback and selected original RGB on both sides."""
    start = perf_counter()
    feedback = read_json(ROOT / case['feedback'])
    read_s = perf_counter() - start
    decode_s = 0.0
    for image in case['images']:
        from PIL import Image
        start = perf_counter()
        raw = (data / case['episode'] / 'raw/task' / image).read_bytes()
        read_s += perf_counter() - start
        start = perf_counter()
        with Image.open(io.BytesIO(raw)) as frame:
            frame.load()
            assert frame.mode == 'RGB'
        decode_s += perf_counter() - start
    # Keep the adjudication available to both callers, outside automated event credit.
    return feedback, {'shared_file_read_and_json_s': read_s, 'rgb_decode_s': decode_s,
                      'decoded_rgb_frames': len(case['images'])}


def real_diagnosis(engine, capture, cache_dir):
    if engine == 'v2':
        from robot_agent.real_replay import real_replay
        r = real_replay(capture, cache_dir)
        return dict(label=r['native_success'], event=r['recorded_stop'], root=r['root_cause_status'],
                    next_check=None, decision=r['decision'], cache_hit=r['cache_hit'])
    from evidence_cache import EvidenceCache
    from real_episode_evidence import summarize_capture
    cache = EvidenceCache(cache_dir)
    try:
        result = summarize_capture(capture, cache)
    finally:
        cache.close()
    r = result['value']
    return dict(label=r['physical_task_success'], event=r['recorded_stop_event'], root=r['root_cause'],
                next_check=r['next_observation'], decision=None, cache_hit=result['cache']['hit'])


def sim_diagnosis(engine, path):
    if engine == 'v2':
        from triage_episode import replay
        r = replay(path)
        d = r['diagnosis']
        event = d['first_divergence'] or {}
        answer = r['decision']
        assert not r['model_invoked']
        return dict(label=d['task_success'], status=d['status'], event=event.get('kind'),
                    source=event.get('sensor_provenance'), phase=event.get('phase'),
                    frame=event.get('first_observed_frame'), root=d['root_cause_status'],
                    decision=answer['decision'], next_check=answer.get('checks'),
                    vetoes=answer.get('capsule', {}).get('counterexamples', []),
                    frame_count=r['measured_frames'], candidate=bool(r['graph'] and r['graph']['request']))
    from diagnose_retention import diagnose_episode
    r = read_json(path)
    retention = diagnose_episode(r)
    native = r['observed_effects'][0]
    result = dict(label=r['task_success'], status=retention['status'], phase=native['last_phase'],
                  native_error=(native.get('error') or '').split('[')[0].strip() or None,
                  frame=retention.get('earliest_observed_frame'), root='UNKNOWN')
    if not r['task_success']:
        from agent_loop import repair_diagnosis
        from agent_challenge import precontact_observation
        # The selected failures end at align/fold. This legacy path never starts a new probe.
        assert native['last_phase'] in ('align', 'fold')
        context = repair_diagnosis(r)
        precontact = precontact_observation(r['experiment_id'])
        result.update(precontact_xy_mm=precontact.get('observed_precontact_xy_error_mm'),
                      memory_ids=[m['id'] for m in context['prior_failure_warnings']],
                      next_check=context.get('next_unknown'), rejected_tilts=context['rejected_grasp_tilt_deg'])
    return result


def worker(case, engine, cache_dir):
    from learn import DATA
    start = perf_counter()
    feedback, cost = shared_feedback(case, DATA)
    assert feedback
    t = perf_counter()
    if case['domain'] == 'real':
        output = real_diagnosis(engine, DATA / case['episode'] / 'raw/task', cache_dir)
    else:
        output = sim_diagnosis(engine, ROOT / case['result'])
    cost.update(engine_s=perf_counter()-t, worker_s=perf_counter()-start)
    return dict(output=output, cost=cost)


def process_pair(case, engine, cache_dir):
    args = [sys.executable, __file__, '--worker', case['id'], engine, str(cache_dir)]
    start = perf_counter()
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        cold = json.loads(proc.stdout.readline())
        cold['cost']['end_to_end_s'] = perf_counter()-start
        start = perf_counter()
        proc.stdin.write('hot\n')
        proc.stdin.flush()
        hot = json.loads(proc.stdout.readline())
        hot['cost']['end_to_end_s'] = perf_counter()-start
        if proc.wait() != 0:
            raise RuntimeError('Benchmark worker failed')
    for run in (cold, hot):
        assert run['output']['label'] is case['success']
    if case['domain'] == 'real':
        assert cold['output']['cache_hit'] is False and hot['output']['cache_hit'] is True
    def stable(run):
        return {k:v for k,v in run['output'].items() if k != 'cache_hit'}
    assert stable(cold) == stable(hot)
    return dict(output=stable(cold), cold=cold['cost'], hot=hot['cost'])


def compare(cases):
    from learn import DATA, digest
    # Verify input pins outside timed runs. This can warm the OS page cache.
    for case in cases:
        for root, pins in [(ROOT,case['repo_pins']), (DATA,case['data_pins'])]:
            for name, expected in pins.items():
                if digest(root / name) != expected:
                    raise ValueError('Frozen input changed: ' + name)
    measurements = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, case in enumerate(cases):
            row = dict(id=case['id'], stratum=case['stratum'], domain=case['domain'])
            order = ['legacy','v2'] if i % 2 == 0 else ['v2','legacy']
            for engine in order:
                row[engine] = process_pair(case, engine, Path(tmp) / case['id'] / engine)
            measurements.append(row)
    return measurements


if __name__ == '__main__':
    design = read_json(HERE / 'design.json')
    if len(sys.argv) == 1:
        print(json.dumps(compare(design['cases']), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        mode, ident, engine, cache_dir = sys.argv[1:]
        assert mode == '--worker' and engine in ('legacy','v2')
        case, = (c for c in design['cases'] if c['id'] == ident)
        print(json.dumps(worker(case, engine, cache_dir), allow_nan=False), flush=True)
        if input() == 'hot':
            print(json.dumps(worker(case, engine, cache_dir), allow_nan=False), flush=True)

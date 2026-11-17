"""Offline archived-evidence replay. Zero new robot/simulation episodes and zero model calls."""
import json
import time
from pathlib import Path
from evidence_cache import EvidenceCache
from episode_evidence import locate, json_rows, summary, summarize
from contact_evidence import inspect_frame
from learn import ROOT, record, finish, save

EPISODES=['20261116T114608Z_4d3972ba','20261116T125354Z_b5546f36','20261116T125834Z_58df2069']


def main():
    start=time.monotonic()
    out,r=record('goal_aligned_runtime','Archived summary/contact tool latency and evidence fidelity','offline_saved_simulation')
    record_seconds=time.monotonic()-start
    cache=EvidenceCache(ROOT/'.local'/r['experiment_id']/'cache')
    results=[]
    r.update(task_success=None,tokens=0,llm_inference_seconds=0,simulation_steps=0,
        privileged_information_used=True,model_calls=0,new_episodes=0,research_orchestrator_tokens=None,
        initial_state=dict(episodes=EPISODES,execution='offline recorded state only'),
        expected_effects=['Same summary and contact results on cache hit; no historical outcome changes'],
        next_improvement='Run authorized model-driven feedback and held-out physical simulation before default promotion')
    try:
        for ident in EPISODES:
            report,episode,raw=locate(ident)
            t=time.monotonic()
            source=json.loads(report.read_text())
            rows=json_rows(raw/'trajectory.jsonl');extras=json_rows(episode/'proprio_stats/privileged.jsonl')
            baseline=summarize(source,rows,extras)
            uncached_seconds=time.monotonic()-t
            cold=summary(ident,cache);warm=summary(ident,cache)
            assert cold['value']==warm['value']==baseline, 'Cold/warm/uncached summary must be identical JSON values'
            assert not cold['cache']['hit'] and warm['cache']['hit'] and warm['cache']['bytes_hashed']==0
            frame=next(x['frame'] for x in baseline['phases'] if x['phase']=='grasp')
            contact=inspect_frame(ident,frame,cache);reused=inspect_frame(ident,frame,cache)
            assert contact['value']==reused['value'] and reused['cache']['hit'], 'Contact cache changed the result'
            assert reused['cache']['bytes_hashed']==0
            save(out/(ident+'_contact.json'),contact)
            results.append(dict(episode_id=ident,physical_outcome=source['task_success'],frame=frame,
                summary_equal=True,contact_equal=True,uncached_summary_seconds=uncached_seconds,
                summary_cold=cold['cache'],summary_warm=warm['cache'],
                contact_cold=contact['cache'],contact_warm=reused['cache'],
                compact_summary_bytes=len(json.dumps(baseline).encode()),
                source_trace_bytes=(raw/'trajectory.jsonl').stat().st_size+(episode/'proprio_stats/privileged.jsonl').stat().st_size,
                retention_status=baseline['retention']['status'],contact_state_source=contact['value']['state_source']))
        r.update(result='success',validation_status='OFFLINE_TOOL_VERIFIED',root_cause='UNKNOWN',
            lessons_learned=['Caching reproduces analysis, not new physical capability.',
                'Historical force estimates are reconstructed and cannot identify real gripper failure.',
                'Original source program, physical labels, speed and safety rules were not changed.'])
    except Exception as exc:
        r.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        cache.close()
        r.update(experiments=results,execution_time=time.monotonic()-start,
                 timing=dict(record_version_snapshot_seconds=record_seconds),
                 limitations=['Three existing Episodes, not a random sample or task-success benchmark',
                    'Warm cache measurements include filesystem/stat and local database costs; no video decoding',
                    'Per-call costs exclude research, publication, model calls and new physical collection',
                    'Default Agent remains unchanged pending actual model/physical comparison'])
        finish(out,r)
    return r


if __name__=='__main__':main()

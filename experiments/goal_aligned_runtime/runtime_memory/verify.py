"""Offline replay of retrieval on archived Episodes; no model, simulator or hardware calls."""
import json
from pathlib import Path
import subprocess
import sys
import time
import types
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import learn

PARENT='c5708a008393446f15267a2c60f0bd3f4b27182c'
EPISODES=['20261117T071047Z_e43b1ad6','20261117T073201Z_efc6139b','20261117T072218Z_43239686']


def main():
    parent=types.ModuleType('parent_learn');parent.__file__=str(ROOT/'scripts/learn.py')
    exec(subprocess.check_output(['git','show',PARENT+':scripts/learn.py'],cwd=ROOT,text=True),parent.__dict__)
    rows=[]
    for ident in EPISODES:
        path=next((ROOT/'experiments').glob('*/'+ident+'/result.json'))
        episode=json.loads(path.read_text())
        started=time.monotonic();old=parent.failure_memory(episode['observed_effects']);old_time=time.monotonic()-started
        started=time.monotonic();new=learn.episode_memory(episode);new_time=time.monotonic()-started
        rows.append(dict(episode_id=ident,source=str(path.relative_to(ROOT)),sha256=learn.digest(path),
            target=episode['initial_state']['target_index'],
            parent_unscoped_ids=[m['id'] for m in old],
            parent_default_right_ids=[],parent_default_basis='Right-arm branch passed no failure memories',
            candidate_ids=[m['id'] for m in new],
            candidate_cards=new,candidate_utf8_bytes=len(json.dumps(new,ensure_ascii=False,separators=(',',':')).encode()),
            parent_lookup_seconds=old_time,candidate_lookup_seconds=new_time))
    assert all('NE020' in r['candidate_ids'] for r in rows[:2])
    assert 'NE020' not in rows[2]['candidate_ids']
    result=dict(experiment_id='offline_runtime_memory_20261117',timestamp=datetime.now(timezone.utc).isoformat(),
        parent_commit=PARENT,validation_status='OFFLINE_RETRIEVAL_AND_ENTRY_BOUNDARY_VERIFIED',
        new_model_calls=0,new_model_tokens=0,physics_steps=0,hardware_commands=0,
        source_sha256={p:learn.digest(ROOT/p) for p in ('scripts/learn.py','scripts/agent_loop.py','scripts/bounded_tools.py','scripts/benchmark_recipient_feedback.py')},
        rows=rows,tests='tests/test_runtime_memory.py',
        limitations=['One lookup per archived Episode, no latency speedup claim.',
            'Parent lookup runs against the same current memory corpus to isolate ranking/filtering; its actual default right route supplied no memory.',
            'Entry test intercepts the model boundary; no live model or robot outcome is attributed to this change.',
            'Two source-linked cases are development regression cases, not independent task generalization.',
            'No reproduction of AWM or full SKILL.state, and no physical repair accepted.'])
    learn.save(Path(__file__).with_name('result.json'),result)
    print(json.dumps([dict(episode=r['episode_id'],before=r['parent_default_right_ids'],after=r['candidate_ids'],bytes=r['candidate_utf8_bytes'],seconds=r['candidate_lookup_seconds']) for r in rows]))


if __name__=='__main__':main()

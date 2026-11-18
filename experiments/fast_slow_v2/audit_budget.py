"""Reconcile completed rounds and consumed native slots, including interrupted starts."""
import json
from pathlib import Path
here=Path(__file__).resolve().parent


def audit():
    budget=json.loads((here.parent/'round_budget.json').read_text())
    totals={s:dict(rounds=0,simulation_starts=0,complete_recordings=0,slow_calls=0,input_tokens=0,output_tokens=0)
            for s in budget['stages']}
    for spec in budget['schedule']:
        i=spec['round_id']
        folder=here/i
        stage='B' if i=='B1' else i.split('.')[0]
        counts=totals[stage]
        starts=list(folder.glob('*_start.json'))
        counts['simulation_starts']+=len(starts)
        for start in starts:
            result=start.with_name(start.name.replace('_start.json','_result.json'))
            if result.exists() and json.loads(result.read_text()).get('recording_verified'):
                counts['complete_recordings']+=1
        if i in budget['completed']:
            r=json.loads((folder/'RESULT.json').read_text())
            assert r['sim_episode_count']==len(starts) and r['llm_calls']<=budget['per_round']['slow_calls']
            assert r['rollback_ref'] and r['source_hash'] and r['test_count']>0 and r['hypothesis']==spec['hypothesis']
            counts['rounds']+=1
            for field in ('input_tokens','output_tokens'):
                counts[field]+=r[field]
            counts['slow_calls']+=r['llm_calls']
    for stage,counts in totals.items():
        expected=budget['stages'][stage]
        assert counts['rounds']==expected['used_rounds']<=expected['round_cap']
        assert counts['simulation_starts']==expected['sim_used']<=expected['sim_cap']
    assert sum(x['rounds'] for x in totals.values())+budget['remaining_rounds']==14
    assert sum(x['simulation_starts'] for x in totals.values())+budget['remaining_simulations']==24
    return totals


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))

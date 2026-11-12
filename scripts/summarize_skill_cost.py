"""Evaluate the frozen Stage4 skill gates and activate or retain the planner accordingly."""
import json
from datetime import datetime, timezone
import numpy as np
from learn import ROOT, DATA, save, digest


def main():
    folder=ROOT/'experiments/research_stage4/round4'
    ids=json.loads((folder/'progress.json').read_text())['experiments']
    assert len(ids)==4, 'Do not promote a partial comparison'
    rows=[]
    for ident in ids:
        r=json.loads((ROOT/'experiments/research_stage4'/ident/'result.json').read_text())
        t=json.loads((ROOT/'experiments/research_stage4'/r['experiments'][0]['experiment_id']/'result.json').read_text())
        trajectory=[json.loads(x) for x in (DATA/'self_evolution'/t['experiment_id']/'trajectory.jsonl').read_text().splitlines()]
        q=np.rad2deg(np.array([x['q'] for x in trajectory])[:,[0,1,2,3,4,5,7,8,9,10,11,12]])
        dt=np.diff([x['time'] for x in trajectory]);velocity=np.diff(q,axis=0)/dt[:,None]
        quality=dict(peak_joint_speed_deg_s=float(abs(velocity).max()),peak_joint_acceleration_deg_s2=float(abs(np.diff(velocity,axis=0)/dt[1:,None]).max()),joint_path_sum_deg=float(abs(np.diff(q,axis=0)).sum()))
        rows.append(dict(experiment_id=ident,physical_trial_id=t['experiment_id'],
            candidate=r['observation_variant']['planner_policy']=='compiled_alignment_v1',
            actual_planner=r['actual_planner_kind'],model_invoked=r['model_invoked'],model_requested=r['model'],
            shift_mm=r['initial_state']['case']['shift_mm'],prompt_sha256=r['prompt_sha256'],
            task_config_sha256=t['initial_state']['task_config_sha256'],
            skill_router_sha256=r['skill_router_sha256'],offset_mm=r['actions'][0]['offset_mm'],
            task_success=t['task_success'],valid_successful_episode=r['valid_successful_episode'],
            recording_verified=t['recording_verified'],causal_diagnosis=t['causal_diagnosis'],
            tokens=r['tokens'],planning_seconds=r['planning_seconds'],llm_wall_seconds=r['llm_inference_seconds'],
            context_preparation_seconds=r['context_preparation_seconds'],
            collection_plus_planning_seconds=r['collection_plus_planning_seconds'],
            simulation_seconds=t['simulation_seconds'],quality_30hz=quality,
            source_unchanged=t['source_scripts_src_configs_unchanged'],tool_events=r['planner_calls'][0]['tool_events']))
    pairs=[]
    for shift in [.75,-.75]:
        baseline=next(x for x in rows if x['shift_mm']==shift and not x['candidate'])
        candidate=next(x for x in rows if x['shift_mm']==shift and x['candidate'])
        same=baseline['prompt_sha256']==candidate['prompt_sha256'] and baseline['task_config_sha256']==candidate['task_config_sha256']
        quality=(candidate['quality_30hz']['peak_joint_speed_deg_s']<=12 and candidate['simulation_seconds']<=1.05*baseline['simulation_seconds'] and all(candidate['quality_30hz'][k]<=1.10*baseline['quality_30hz'][k] for k in ['peak_joint_acceleration_deg_s2','joint_path_sum_deg']))
        pair_ok=(same and quality and candidate['valid_successful_episode'] and candidate['actual_planner']=='compiled_alignment_v1' and not candidate['model_invoked'] and candidate['recording_verified'] and candidate['source_unchanged'] and candidate['tool_events']==0)
        pairs.append(dict(shift_mm=shift,same_facts_and_initial_config=same,quality_gate=quality,accepted=pair_ok))
    metrics={}
    for flag in [False,True]:
        batch=[x for x in rows if x['candidate']==flag];n=sum(x['valid_successful_episode'] for x in batch)
        tokens=sum(x['tokens']['input_tokens']+x['tokens']['output_tokens'] for x in batch)
        seconds=sum(x['collection_plus_planning_seconds'] for x in batch)
        metrics['candidate' if flag else 'baseline']=dict(valid_successes=n,attempts=len(batch),success_rate=n/len(batch),model_calls=sum(x['model_invoked'] for x in batch),reported_input_plus_output_tokens=tokens,component_seconds=seconds,tokens_per_valid_success=tokens/n if n else None,seconds_per_valid_success=seconds/n if n else None,valid_successes_per_component_hour=n*3600/seconds)
    token_gate=metrics['candidate']['reported_input_plus_output_tokens']<=.2*metrics['baseline']['reported_input_plus_output_tokens']
    current_hash=digest(ROOT/'scripts/alignment_skill.py')
    accepted=all(x['accepted'] for x in pairs) and token_gate and all(x['skill_router_sha256']==current_hash for x in rows)
    now=datetime.now(timezone.utc).isoformat()
    result=dict(stage=4,round=4,timestamp=now,result='ACCEPTED_SIMULATION_KNOWN_SKILL_ONLY' if accepted else 'REJECTED_KEEP_LLM',accepted=accepted,runs=rows,pairs=pairs,metrics=metrics,
        planner_token_reduction_fraction=1-metrics['candidate']['reported_input_plus_output_tokens']/metrics['baseline']['reported_input_plus_output_tokens'],
        component_seconds_reduction_fraction=1-metrics['candidate']['component_seconds']/metrics['baseline']['component_seconds'],
        privileged_information_used=True,privileged_fields_used=['precontact object/socket residual','coordinate transform','finger-object contact','object/gripper state','native task success and trajectory'],
        task_family='Existing plug insertion and one-axis alignment repair',new_skill_discovery_count=0,existing_skill_compiled_count=1,
        new_episodes={'success':sum(x['valid_successful_episode'] for x in rows),'failure':sum(not x['task_success'] for x in rows)},
        human_interventions=None,research_orchestrator_tokens=None,monetary_cost=None,new_real_motion=False,
        limitations=['N=2 matched pairs, not sustained-production throughput','No unknown task-family or weak-model reasoning improvement demonstrated','Planner tokens exclude research/CLI orchestration overhead and are not billed monetary cost','Simulator remains unvalidated against independent real calibration data'])
    save(folder/'result.json',result)
    manifest_path=ROOT/'evolution/versions/compiled_alignment_v1.json'
    manifest=json.loads(manifest_path.read_text());manifest.update(accepted=accepted,status='ACTIVE_AGENT_D_SIMULATION_SCOPE' if accepted else 'REJECTED',implementation_sha256=current_hash,evidence='experiments/research_stage4/round4/result.json',timestamp=now);save(manifest_path,manifest)
    active_path=ROOT/'evolution/active.json';active=json.loads(active_path.read_text())
    active['previous_planner_policy']=active.get('planner_policy','llm')
    active['planner_policy']='compiled_alignment_v1' if accepted else 'llm'
    active['planner_evidence']='experiments/research_stage4/round4/result.json'
    save(active_path,active)
    lines=['# Stage4 第四轮：可执行经验与重复模型推理的完整采集对照','',f'判定：**{result["result"]}**。两个预先固定的初始条件、每组同一份事实与一次执行预算；控制、物理、时序、成功检测、录像和诊断相同。','', '| 路径 | 有效成功 | 规划模型调用 | 输入+输出Token/成功 | 组件秒/成功 | 成功/组件小时 |','|---|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():lines.append(f'| {name} | {m["valid_successes"]}/2 | {m["model_calls"]} | {m["tokens_per_valid_success"]} | {m["seconds_per_valid_success"]} | {m["valid_successes_per_component_hour"]:.3f} |')
    lines+=['',f'记录到的规划Token减少{result["planner_token_reduction_fraction"]:.1%}，这组完整采集组件时间减少{result["component_seconds_reduction_fraction"]:.1%}。包含两组上下文准备时间；不含研究、Git、空闲、人工和收费金额。候选没有调用LLM不代表总研究成本为零。','', '规则按当前失败观测计算 previous_offset−residual_X，不存储任务专用正确答案。来源是已验证VS002；输出仍交给现有参数边界和物理任务检查。来源/目标/坐标/单位/接触条件不适用时回到原LLM路径，不向真机推广。','', '既有agent_loop D通过同一个plan_with_memory函数读取active.json.planner_policy，并调用完整记录器；A/B/C继续模型规划。报告保存actual_planner_kind和model_invoked，已验证路由与数据调用的单元测试。旧negative-experience version v2不变。回滚使用--planner-policy llm或将planner_policy改为llm。','', '四条完整Episode与预定运动质量门槛逐条保存。Simulation-only，使用特权位姿/接触/坐标/任务真值。新初始条件不等于新任务；这是已有经验程序化复用，不是弱模型学会未知任务，也不是完整自进化或真实数字孪生校准的证明。']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (ROOT/'evolution/evolution_log.jsonl').open('a') as f:f.write(json.dumps(dict(timestamp=now,stage=4,round=4,candidate='compiled_alignment_v1',accepted=accepted,active_planner=active['planner_policy'],evidence='experiments/research_stage4/round4/result.json'))+'\n')
    print(json.dumps(dict(accepted=accepted,metrics=metrics,pairs=pairs),indent=2))


if __name__=='__main__':main()

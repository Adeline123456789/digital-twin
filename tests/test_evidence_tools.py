import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mujoco
import numpy as np
from evidence_cache import EvidenceCache
from episode_evidence import compact_frame
from contact_evidence import integration_snapshot, inspect_state
from bounded_tools import plan, total_usage, baseline_case, validate_tool_plan, prompt_events
from model_compare import ask, ModelCallError


class EvidenceTests(unittest.TestCase):
    def test_summary_contact_pairs_survive_json_roundtrip(self):
        row=dict(phase='grasp',time=1,plug=[0,0,0],grip=[0,0,.02])
        extra=dict(contacts=[dict(body1='task/yellow_plug',body2='right/link7')])
        value=compact_frame(row,extra,0)
        self.assertEqual(value,json.loads(json.dumps(value)))

    def test_content_cache_invalidates_changed_dependency_and_reuses_old_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'trace';p.write_text('old')
            cache=EvidenceCache(root/'cache');computations=[]
            def get(parameter=1):
                return cache.get('analyzer',{'trace':p},{'threshold':parameter},
                                 lambda:computations.append(p.read_text()) or {'text':p.read_text()})
            a=get();b=get();p.write_text('new');c=get();p.write_text('old');d=get();e=get(2)
            self.assertEqual([x['cache']['hit'] for x in (a,b,c,d,e)],[False,True,False,True,False])
            self.assertEqual(b['cache']['bytes_hashed'],0)
            self.assertEqual(c['value']['text'],'new');self.assertEqual(d['value']['text'],'old')
            self.assertEqual(len(computations),3);cache.close()

    def test_inactive_margin_contact_is_not_gripping_force(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="1 1 .1" margin=".02" gap=".02"/>
          <body name="task/yellow_plug" pos="0 0 .11"><freejoint/>
          <geom type="sphere" size=".1" mass="1" margin=".02" gap=".02"/></body>
        </worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        result=inspect_state(model,data,surfaces=False)
        self.assertGreater(len(result['contacts']),0)
        self.assertTrue(all(not c['solver_active'] for c in result['contacts']))
        self.assertTrue(all(c['solved_force_contact_frame_N_Nm']==[0.]*6 for c in result['contacts']))

    def test_integration_state_preserves_applied_force_and_time(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model);data.xfrc_applied[1,0]=2.;data.time=3.;data.qvel[0]=.1
        saved=integration_snapshot(model,data);other=mujoco.MjData(model)
        mujoco.mj_setState(model,other,np.array(saved['values']),saved['spec'])
        self.assertEqual(other.time,3.);np.testing.assert_array_equal(other.xfrc_applied,data.xfrc_applied)

    def test_model_timeout_keeps_available_usage_and_no_fabricated_answer(self):
        import subprocess
        output=json.dumps({'type':'turn.completed','usage':{'input_tokens':12,'output_tokens':3}}).encode()
        with patch('model_compare.subprocess.run',side_effect=subprocess.TimeoutExpired('codex',1,output=output)):
            with self.assertRaises(ModelCallError) as caught:ask('x',Path('/tmp'),timeout=1)
        self.assertTrue(caught.exception.call['timed_out'])
        self.assertEqual(caught.exception.call['usage']['input_tokens'],12)
        self.assertIsNone(caught.exception.call['answer'])
        self.assertIsNone(total_usage([{'usage':None}]))


class FakeTools:
    context={'baseline':{'offset_mm':0,'grasp_tilt_deg':0},'allowed_grasp_tilt_deg':[]}
    def __init__(self):self.requests=[]
    def call(self,name,args):
        if name!='episode_summary' or args:raise ValueError('Unknown tool')
        self.requests.append(name)
        return {'value':{'retention':'UNKNOWN'},'cache':{'hit':True,'key':'fixture'}}


class RouterTests(unittest.TestCase):
    def test_initial_summary_is_actual_observation_without_an_extra_model_call(self):
        tools=FakeTools();prompts=[]
        def model(prompt,*args,**kwargs):
            prompts.append(prompt)
            return dict(answer={'plan':{'stop':True,'hypotheses':[]}},usage={'input_tokens':3},wall_seconds=.1,tool_events=0)
        result=plan(self.seed,Path('/tmp'),ask_fn=model,dispatcher=tools,max_calls=1,initial_summary=True)
        self.assertEqual(tools.requests,['episode_summary'])
        self.assertEqual(len(prompts),1)
        self.assertIn('"observed_summary":{"retention":"UNKNOWN"}',prompts[0])
        self.assertEqual(result['tool_trace'],[])
        self.assertEqual(result['initial_observation']['result']['value'],{'retention':'UNKNOWN'})

    def test_prompt_omits_dependency_hash_manifest_but_keeps_evidence(self):
        original={'request':{'tool':'episode_summary'},'result':{'value':{'root_cause':'UNKNOWN'},
                  'cache':{'key':'abc','dependency_sha256':{'large':'manifest'}}}}
        reduced=prompt_events([original])
        self.assertEqual(reduced[0]['value'],original['result']['value'])
        self.assertNotIn('dependency_sha256',json.dumps(reduced))
        self.assertIn('dependency_sha256',original['result']['cache'])

    seed=dict(experiment_id='fixture',task='test',initial_state={},validation_status='UNVERIFIED',task_success=False)
    def run_requests(self,requests,tools=None):
        seen=[]
        def model(prompt,*args,**kwargs):
            seen.append(prompt)
            return dict(answer=requests[len(seen)-1],usage={'input_tokens':3,'output_tokens':2},wall_seconds=.1,tool_events=0)
        result=plan(self.seed,Path('/tmp'),ask_fn=model,dispatcher=tools or FakeTools(),max_calls=len(requests))
        return result,seen

    def test_actual_tool_feedback_is_in_next_model_input(self):
        tools=FakeTools()
        r,prompts=self.run_requests([{'tool':'episode_summary','args':{},'reason':'inspect'},
            {'plan':{'offset_mm':1,'hypotheses':[],'reason':'candidate'}}],tools)
        self.assertEqual(tools.requests,['episode_summary'])
        self.assertIn('retention',prompts[1]);self.assertEqual(r['status'],'PLAN_READY')
        self.assertEqual(r['usage']['input_tokens'],6)

    def test_unknown_shell_and_duplicate_requests_never_dispatch_twice(self):
        tools=FakeTools();request={'tool':'episode_summary','args':{},'reason':'inspect'}
        r,_=self.run_requests([request,request],tools)
        self.assertEqual(tools.requests,['episode_summary']);self.assertTrue(r['answer']['stop'])
        self.assertEqual(r['status'],'DUPLICATE_TOOL_REJECTED')
        r,_=self.run_requests([{'tool':'shell','args':{'cmd':'move'},'reason':'unsafe'}],tools)
        self.assertEqual(r['status'],'EVIDENCE_OR_REQUEST_ERROR');self.assertTrue(r['answer']['stop'])

    def test_existing_entry_routes_tool_policy_and_preserves_parameter_baseline(self):
        import agent_loop, benchmark_collection
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for d in ['experiments/baseline/20261111T083720Z_f64309a8','experiments/evaluation/seed']:
                (root/d).mkdir(parents=True)
            seed=dict(experiment_id='seed',active_arms=['right'],initial_state={'target_index':3},actions=[],observed_effects=[])
            for rel in ['baseline/20261111T083720Z_f64309a8','evaluation/seed']:
                (root/'experiments'/rel/'result.json').write_text(json.dumps(seed))
            context={'baseline':{'offset_mm':0,'grasp_tilt_deg':0,'grasp_height_m':.016,'align_gap_m':.023},'allowed_grasp_tilt_deg':[]}
            call=dict(answer={'offset_mm':1,'hypotheses':[]},applicability=context,usage={'input_tokens':5},tool_events=0,
                      status='PLAN_READY',model_calls_attempted=1,tool_trace=[])
            trial=dict(experiment_id='physics',task_success=False,result='failure',observed_effects=[],execution_time=1,
                       collection_component_seconds=1,recording_verified=True)
            with patch.object(agent_loop,'ROOT',root),patch.object(agent_loop,'record',return_value=(root,{'experiment_id':'candidate'})),patch.object(agent_loop,'finish'),patch('bounded_tools.plan',return_value=call) as planner,patch.object(benchmark_collection,'run_case',return_value=trial) as collect,patch('episode_format.source_for'),patch('agent_challenge.execution_scene',return_value={'fixture_scene':True}),patch.dict(agent_loop.os.environ,{'PIPER_ACTIVE_ARMS':'right','PIPER_SIM_PROFILE':'original','PIPER_TARGET_INDEX':'3'}):
                result=agent_loop.run('D',0,False,'bounded_tools_v1','seed',max_executions=1)
            planner.assert_called_once();collect.assert_called_once()
            self.assertTrue(planner.call_args.kwargs['initial_summary'])
            self.assertEqual(planner.call_args.kwargs['scene'],{'fixture_scene':True})
            self.assertEqual(result['actual_planner_policy'],'bounded_tools_v1')
            self.assertEqual(collect.call_args.args[0]['grasp_height_m'],.016)
            self.assertEqual(collect.call_args.args[0]['align_gap_m'],.023)
            self.assertFalse(result['task_success'])

    def test_unscreened_tilt_cannot_be_executed(self):
        r,_=self.run_requests([{'plan':{'grasp_tilt_deg':5,'hypotheses':[]}}])
        self.assertEqual(r['status'],'EVIDENCE_OR_REQUEST_ERROR');self.assertTrue(r['answer']['stop'])

    def test_agent_entry_unknown_is_not_labelled_a_simulated_failure(self):
        import agent_loop, benchmark_collection
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            seed=dict(experiment_id='seed',active_arms=['right'],initial_state={'target_index':3},actions=[],observed_effects=[])
            for rel in ['baseline/20261111T083720Z_f64309a8','evaluation/seed']:
                path=root/'experiments'/rel/'result.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(seed))
            call=dict(answer={'stop':True,'hypotheses':[]},applicability={},usage={'input_tokens':5},tool_events=0,
                      status='STOPPED_UNKNOWN',model_calls_attempted=1,tool_trace=[])
            with patch.object(agent_loop,'ROOT',root),patch.object(agent_loop,'record',return_value=(root,{'experiment_id':'candidate'})),patch.object(agent_loop,'finish'),patch('bounded_tools.plan',return_value=call),patch.object(benchmark_collection,'run_case') as collect,patch('episode_format.source_for'),patch('agent_challenge.execution_scene',return_value={'fixture_scene':True}),patch.dict(agent_loop.os.environ,{'PIPER_ACTIVE_ARMS':'right','PIPER_SIM_PROFILE':'original','PIPER_TARGET_INDEX':'3'}):
                result=agent_loop.run('D',0,False,'bounded_tools_v1','seed',max_executions=1)
            collect.assert_not_called()
            self.assertIsNone(result['task_success'])
            self.assertEqual(result['validation_status'],'NO_PHYSICAL_EXECUTION')

    def test_one_parameter_repair_preserves_recorded_height_gap_and_start(self):
        seed=dict(actions=[{'offset_mm':[0,0],'grasp_height_m':.016}],initial_state={},
                  observed_effects=[{'grasp_tilt_deg':0,'align_gap_m':.023}],
                  comparison_case={'start_right_deg':[0,0,0,0,20,0]})
        context=dict(baseline=baseline_case(seed),allowed_grasp_tilt_deg=[])
        case=validate_tool_plan(dict(offset_mm=1,hypotheses=[]),context)
        self.assertEqual(case['grasp_height_m'],.016);self.assertEqual(case['align_gap_m'],.023)
        self.assertEqual(case['start_right_deg'],[0,0,0,0,20,0])
        changed=validate_tool_plan(dict(align_gap_m=.024,hypotheses=[]),context)
        self.assertEqual(changed['grasp_height_m'],.016);self.assertEqual(changed['offset_mm'],0)
        with self.assertRaises(ValueError):validate_tool_plan(dict(align_gap_m=.024,offset_mm=1,hypotheses=[]),context)


if __name__=='__main__':unittest.main()

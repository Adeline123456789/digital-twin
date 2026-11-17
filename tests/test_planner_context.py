import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from model_compare import ask


class PlannerContextTests(unittest.TestCase):
    def test_only_catalog_flag_changes_and_usage_is_preserved(self):
        events = [dict(type='item.completed', item=dict(type='agent_message', text='{"offset_mm":0}')),
                  dict(type='turn.completed', usage=dict(input_tokens=123, output_tokens=12, cached_input_tokens=0))]
        response = SimpleNamespace(returncode=0, stdout='\n'.join(map(json.dumps, events)), stderr='')
        calls = []
        with patch('model_compare.subprocess.run', side_effect=lambda cmd, **kw: calls.append((cmd, kw)) or response):
            baseline = ask('same physical evidence', Path('/tmp'))
            candidate = ask('same physical evidence', Path('/tmp'), include_catalog=False)
        self.assertEqual(calls[0][0][:-1], calls[1][0][:-3])
        self.assertEqual(calls[1][0][-3:], ['-c', 'skills.include_instructions=false', '-'])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(baseline['usage'], candidate['usage'])
        self.assertEqual(candidate['tool_events'], 0)
        self.assertEqual(candidate['event_type_counts'],{'item.completed':1,'turn.completed':1})
        self.assertEqual(candidate['completed_item_type_counts'],{'agent_message':1})
        self.assertEqual(candidate['prompt_utf8_bytes'],len('same physical evidence'.encode()))
        self.assertEqual(candidate['process_returncode'],0)

    def test_tool_events_are_not_hidden_by_catalog_option(self):
        response = SimpleNamespace(returncode=0, stderr='', stdout='\n'.join(map(json.dumps, [
            dict(type='item.completed', item=dict(type='command_execution', command='pwd')),
            dict(type='item.completed', item=dict(type='agent_message', text='{}'))])))
        with patch('model_compare.subprocess.run', return_value=response):
            self.assertEqual(ask('return JSON', Path('/tmp'), include_catalog=False)['tool_events'], 1)

    def test_bounded_profile_changes_only_instruction_source(self):
        response=SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(type='item.completed',item=dict(type='agent_message',text='{}'))))
        with patch('model_compare.subprocess.run',return_value=response) as run:
            ask('same facts',Path('/tmp'),include_catalog=False,profile='codex_default')
            base=run.call_args
            answer=ask('same facts',Path('/tmp'),include_catalog=False,profile='bounded_json_v1')
            candidate=run.call_args
        self.assertEqual(base.kwargs,candidate.kwargs)
        self.assertEqual(base.args[0][:-1],candidate.args[0][:-3])
        self.assertTrue(candidate.args[0][-2].startswith('model_instructions_file='))
        self.assertEqual(answer['planner_profile'],'bounded_json_v1')
        self.assertIn('read-only',candidate.args[0])

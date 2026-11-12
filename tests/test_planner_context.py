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

    def test_tool_events_are_not_hidden_by_catalog_option(self):
        response = SimpleNamespace(returncode=0, stderr='', stdout='\n'.join(map(json.dumps, [
            dict(type='item.completed', item=dict(type='command_execution', command='pwd')),
            dict(type='item.completed', item=dict(type='agent_message', text='{}'))])))
        with patch('model_compare.subprocess.run', return_value=response):
            self.assertEqual(ask('return JSON', Path('/tmp'), include_catalog=False)['tool_events'], 1)

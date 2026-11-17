import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from model_compare import ask, ModelCallError


class ModelCompletionTests(unittest.TestCase):
    message = {'type':'item.completed','item':{'type':'agent_message',
               'text':'{"plan":{"stop":true,"hypotheses":[],"reason":"fixture"}}'}}
    completed = {'type':'turn.completed','usage':{'input_tokens':12,'output_tokens':5}}

    def test_completed_turn_survives_actual_child_exit_timeout(self):
        real_run = subprocess.run
        payload = '\n'.join(json.dumps(e) for e in [self.message,self.completed])
        code = 'import sys,time; print('+repr(payload)+',flush=True); time.sleep(5)'
        def child(command, **kwargs):
            return real_run([sys.executable,'-c',code], **kwargs)
        with tempfile.TemporaryDirectory() as folder, patch('model_compare.subprocess.run',side_effect=child):
            call = ask('offline fixture',Path(folder),model='offline-fixture',timeout=.2)
        self.assertTrue(call['answer']['plan']['stop'])
        self.assertTrue(call['timed_out'])
        self.assertTrue(call['completed_turn_before_process_timeout'])
        self.assertEqual(call['usage'],self.completed['usage'])
        self.assertIsNone(call['process_returncode'])

    def test_partial_or_failed_turn_cannot_be_recovered(self):
        for events in [[self.message], [self.message,self.completed,{'type':'turn.failed'}]]:
            output='\n'.join(json.dumps(e) for e in events).encode()
            with self.subTest(events=events), patch('model_compare.subprocess.run',
                    side_effect=subprocess.TimeoutExpired('codex',.2,output=output)):
                with self.assertRaises(ModelCallError) as caught:
                    ask('offline fixture',Path('/tmp'),model='offline-fixture',timeout=.2)
            self.assertIsNone(caught.exception.call['answer'])
            self.assertEqual(caught.exception.call['terminal_agent_message'],self.message['item']['text'])

    def test_native_tool_events_remain_visible_to_the_callers(self):
        events=[{'type':'item.completed','item':{'type':'command_execution'}},self.message,self.completed]
        output='\n'.join(json.dumps(e) for e in events).encode()
        with patch('model_compare.subprocess.run',side_effect=subprocess.TimeoutExpired('codex',.2,output=output)):
            call=ask('offline fixture',Path('/tmp'),model='offline-fixture',timeout=.2)
        self.assertEqual(call['tool_events'],1)


if __name__=='__main__':unittest.main()

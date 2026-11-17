import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import learn
import agent_challenge


class SceneContextTests(unittest.TestCase):
    def test_archived_camera_copy_is_same_dependency_but_changed_contents_are_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'self_evolution/seed';raw.mkdir(parents=True)
            old=raw/'cameras.json';new=root/'cameras.json'
            old.write_text('{"version":1}');new.write_text(old.read_text())
            before=dict(holder_xy_m=[.3,-.3],target_index=3,active_arms=['right'],camera_calibration=str(old))
            after=dict(before,camera_calibration=str(new));path=root/'task.yaml'
            (raw/'task.yaml').write_text(yaml.safe_dump(before));path.write_text(yaml.safe_dump(after))
            with patch.object(agent_challenge,'DATA',root),patch.dict(os.environ,{'PIPER_SIM_PROFILE':'original','PIPER_TASK_CONFIG':str(path),'PIPER_ACTIVE_ARMS':'right','PIPER_TARGET_INDEX':'3'}):
                self.assertEqual(agent_challenge.execution_scene('seed',0)['changed_fields'],[])
                new.write_text('{"version":2}')
                self.assertEqual(agent_challenge.execution_scene('seed',0)['changed_fields'],['camera_calibration'])

    def test_config_return_does_not_accumulate_shift_and_explicit_arms_override_saved_right(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'task.yaml'
            path.write_text('holder_xy_m: [0.3, -0.3]\ntarget_index: 3\nactive_arms: [right]\n')
            with patch.dict(os.environ,{'PIPER_SIM_PROFILE':'original','PIPER_TASK_CONFIG':str(path),'PIPER_ACTIVE_ARMS':'right','PIPER_TARGET_INDEX':'3'}):
                original,_=learn.execution_task(0,True)
                changed,_=learn.execution_task(1,True)
                returned,_=learn.execution_task(0,True)
                self.assertEqual(original,returned)
                self.assertAlmostEqual(changed['holder_xy_m'][0]-original['holder_xy_m'][0],.001)
                with patch.dict(os.environ,{'PIPER_ACTIVE_ARMS':'dual'}):
                    dual,_=learn.execution_task(0,True)
                self.assertEqual(dual['active_arms'],['left','right'])
                self.assertEqual(learn.execution_task(0,True)[0],original)
            self.assertEqual(yaml.safe_load(path.read_text())['active_arms'],['right'])

    def test_context_uses_effective_config_and_hides_private_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'self_evolution/seed';raw.mkdir(parents=True)
            base=dict(holder_xy_m=[.3,-.3],strip_xy_m=[.2,.1],strip_yaw_deg=-90,
                      target_index=3,active_arms=['right'],camera_calibration='local-only.json')
            recorded=dict(base,holder_xy_m=[.304,-.3],camera_calibration='previous-local-only.json')
            path=root/'base.yaml';path.write_text(yaml.safe_dump(base));(raw/'task.yaml').write_text(yaml.safe_dump(recorded))
            with patch.object(agent_challenge,'DATA',root),patch.dict(os.environ,{'PIPER_SIM_PROFILE':'original','PIPER_TASK_CONFIG':str(path),'PIPER_ACTIVE_ARMS':'right','PIPER_TARGET_INDEX':'3'}):
                context=agent_challenge.execution_scene('seed',2)
                effective,_=learn.execution_task(2,True)
            self.assertAlmostEqual(context['holder_translation_world_m'][0],-.002)
            self.assertEqual(context['execution_task_sha256'],hashlib.sha256(yaml.safe_dump(effective).encode()).hexdigest())
            self.assertEqual(context['changed_fields'],['camera_calibration','holder_xy_m'])
            self.assertNotIn('camera_calibration',context['changed_values'])
            self.assertNotIn('local-only',str(context))
            self.assertEqual(context['validation_status'],'CONFIG_RESOLVED_NOT_TRAJECTORY_VERIFIED')

    def test_default_dual_representation_does_not_invent_an_embodiment_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'self_evolution/seed';raw.mkdir(parents=True)
            path=raw/'task.yaml';path.write_text('holder_xy_m: [0.3, -0.3]\ntarget_index: 3\n')
            with patch.object(agent_challenge,'DATA',root),patch.dict(os.environ,{'PIPER_SIM_PROFILE':'original','PIPER_TASK_CONFIG':str(path),'PIPER_ACTIVE_ARMS':'dual','PIPER_TARGET_INDEX':'3'}):
                self.assertEqual(agent_challenge.execution_scene('seed',0)['changed_fields'],[])


if __name__=='__main__':unittest.main()

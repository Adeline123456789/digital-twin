import os
import sys
from pathlib import Path
import unittest
import tempfile
import mujoco
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from learn import SIM
sys.path.insert(0,str(SIM/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.backends.arm_layout import ArmLayout, save_recorded_model
from piper_sim.config import load
from piper_sim.contracts import Command


class ArmLayoutTests(unittest.TestCase):
    def test_right_contract_rejects_dual_commands_and_hardware_contract_stays_dual(self):
        layout=ArmLayout(['right'])
        self.assertEqual(layout.names[0],'right.joint1')
        with self.assertRaises(ValueError):layout.command(np.zeros(14),0,0)
        with self.assertRaises(ValueError):Command(np.zeros(7),0,0)

    def test_right_removal_named_state_and_restore_dual(self):
        cfg=yaml.safe_load((SIM/'configs/tasks/real_connector_calibrated.yaml').read_text())
        reference=None
        for arms in (['left','right'],['right'],['right'],['left','right']):
            sim=MujocoBackend(dict(load(),task_scene=dict(cfg,active_arms=arms)))
            try:
                obs=sim.read()
                model=sim.model
                right={name:model.joint('right/'+name).range.copy() for name in ('joint1','joint2','joint3','joint4','joint5','joint6')}
                if reference is None:reference=right
                for name in right:np.testing.assert_array_equal(reference[name],right[name])
                self.assertEqual(len(obs.position),7*len(arms))
                self.assertEqual(model.nu,7*len(arms))
                self.assertEqual(sim.arms,tuple(arms))
                if arms==['right']:
                    self.assertFalse(any(model.body(i).name.startswith('left/') for i in range(model.nbody)))
                    self.assertNotIn('left_wrist',sim.rig['cameras'])
                    self.assertEqual(set(sim.rig['cameras']),{'right_wrist','overhead'})
                    self.assertEqual(model.body('task/strip').jntnum,0)
                    self.assertGreater(model.body('station/vacant_mount_plate').id,0)
                    with tempfile.TemporaryDirectory() as directory:
                        save_recorded_model(model,Path(directory)/'recorded.xml')
                        restored=mujoco.MjModel.from_xml_path(str(Path(directory)/'recorded.xml'))
                        self.assertEqual(restored.nu,7)
                    with self.assertRaises(ValueError):sim.apply(Command(np.zeros(14),0,0))
                q=obs.position.copy();sim.set_observation_state(q)
                np.testing.assert_array_equal(sim.read().position,q)
            finally:sim.close()


if __name__=='__main__':unittest.main()

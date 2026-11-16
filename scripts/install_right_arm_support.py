"""Apply bounded local simulator integration edits, retaining exact originals locally."""
import json
from pathlib import Path
import shutil
from learn import SIM, DATA, ROOT, digest, save

BACKUP=DATA/'self_evolution/right_arm_adapter_v1/native_backup'


def install():
    changes={}
    def edit(rel, pairs):
        path=SIM/rel
        original=path.read_text();updated=original
        for old,new in pairs:
            if old not in updated:raise ValueError('Source contract changed: '+rel+' / '+old[:70])
            updated=updated.replace(old,new)
        backup=BACKUP/rel
        backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists():raise FileExistsError('Already installed; do not overwrite backup')
        shutil.copy2(path,backup)
        path.write_text(updated)
        changes[rel]={'before':digest(backup),'after':digest(path)}
    edit('src/piper_sim/backends/mujoco.py',[
        ('from ..mujoco_asset import resolve_model','from ..mujoco_asset import resolve_model\nfrom .arm_layout import ArmLayout, right_model'),
        ('        self.data = mujoco.MjData(self.model)','        self.layout = ArmLayout(self.task.config.get("active_arms", ARMS) if self.task else ARMS)\n        self.arms = self.layout.arms\n        self.data = mujoco.MjData(self.model)'),
        ('self.model = mujoco.MjModel.from_xml_string(self.task_adapter.compose_model(path, self.task))','xml = self.task_adapter.compose_model(path, self.task)\n            self.model = right_model(xml) if self.task.config.get("active_arms") == ["right"] else mujoco.MjModel.from_xml_string(xml)'),
        ('        if self.task and hasattr(self.task_adapter, "configure_cameras"):', '        if self.arms == ("right",):\n            self.rig["cameras"].pop("left_wrist")\n            self.rig["mounts"].pop("left")\n        if self.task and hasattr(self.task_adapter, "configure_cameras"):'),
        ('        self.enable_cameras = cameras','        self.initial_position = self.initial_position[self.layout.legacy_ids]\n        self.enable_cameras = cameras'),
        ('for side in ARMS','for side in self.arms'),('enumerate(ARMS)','enumerate(self.arms)'),
        ('q = vector(position)','q = self.layout.vector(position)'),
        ('q = vector(command.position)','q = self.layout.vector(command.position)'),
        ('q < LOWER','q < self.layout.lower'),('q > UPPER','q > self.layout.upper'),
        ('return Observation(q,','return self.layout.observation(q,')])
    edit('scripts/try_plug_pick_sim.py',[
        ('        self.seed=np.deg2rad', '        self.bounds=np.array([sim.model.joint(f"{side}/joint{i}").range for i in range(1,7)]).T\n        self.seed=np.deg2rad'),
        ('LOWER[7:13]','self.bounds[0]'),('UPPER[7:13]','self.bounds[1]'),
        ('q=np.array([0,0,0,0,0,0,.060]*2,dtype=float)','q=np.array([0,0,0,0,0,0,.060]*len(sim.arms),dtype=float)\n    right_joints=[sim.layout.names.index(f"right.joint{i}") for i in range(1,7)]\n    right_gripper=sim.layout.names.index("right.gripper_width")\n    joint_indices=[i for i,name in enumerate(sim.layout.names) if not name.endswith("gripper_width")]\n    gripper_indices=[i for i,name in enumerate(sim.layout.names) if name.endswith("gripper_width")]'),
        ('[7:13]','[right_joints]'),('[13]','[right_gripper]'),
        ('[[0,1,2,3,4,5,7,8,9,10,11,12]]','[joint_indices]'),('[[6,13]]','[gripper_indices]'),
        ('Command(command,sim.sim_time,len(history))','sim.layout.command(command,sim.sim_time,len(history))')])
    edit('scripts/collect_plug_contact_probes.py', [('enumerate((\'left\',\'right\'))','enumerate(sim.arms)')])
    edit('src/piper_sim/mixed_recording.py',[
        ('        self.fps, self.domain = fps, domain','        self.cameras = tuple(self.meta.get("camera_channels", CAMERAS))\n        if self.meta.get("active_arms") == ["right"]:\n            self.meta["schema"] = "connector_mixed.v2"\n        self.fps, self.domain = fps, domain'),
        ('set(images) != set(CAMERAS)','set(images) != set(self.cameras)'),
        ('One image from each of the three cameras is required','One image per declared camera is required')])
    dest=SIM/'src/piper_sim/backends/arm_layout.py'
    if dest.exists():raise FileExistsError(dest)
    shutil.copy2(ROOT/'scripts/arm_layout.py',dest)
    changes[str(dest.relative_to(SIM))]={'before':None,'after':digest(dest)}
    save(ROOT/'experiments/right_arm_adaptation/cycle1/local_integration.json',
         {'backup':str(BACKUP.relative_to(DATA)),'changes':changes,'hardware_code_changed':False})
    print('Installed local-only simulator integration:',len(changes),'files')


if __name__=='__main__':install()

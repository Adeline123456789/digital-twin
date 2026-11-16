"""Restrict the existing local replay/home connection boundary to the present right arm."""
from pathlib import Path
import shutil
from learn import DATA,ROOT,digest,save

HARDWARE=Path('/media/midea/F496AC8096AC4548/act_td3/piper_use_right/replay_connector.py')
BACKUP=DATA/'self_evolution/right_arm_adapter_v1/hardware_backup/replay_connector.py'

if __name__=='__main__':
    source=HARDWARE.read_text()
    anchor='        # Imported only for --execute, so dry-run cannot construct a CAN object.'
    guard='''        if self.sides != ['right'] or self.args.can_right != 'can_arm2':
            raise ValueError('Current installation permits only right arm on can_arm2; left arm is absent')
'''
    assert source.count(anchor)==1 and guard not in source
    assert not BACKUP.exists()
    BACKUP.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HARDWARE,BACKUP)
    HARDWARE.write_text(source.replace(anchor,guard+anchor))
    save(ROOT/'experiments/right_arm_adaptation/cycle2/hardware_boundary.json',
         {'path_label':'piper_use_right/replay_connector.py:Hardware.connect','before_sha256':digest(BACKUP),
          'after_sha256':digest(HARDWARE),'backup':str(BACKUP.relative_to(DATA)),
          'scope':'Replay and home shared connection boundary; existing enable, workspace, speed, stop and confirmation checks unchanged',
          'hardware_motion':False,'restore':'Restore the local backup only after current installation permits dual hardware again'})

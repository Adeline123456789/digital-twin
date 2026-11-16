"""Simulation-only compact arm layout; hardware contracts remain unchanged."""
from types import SimpleNamespace
import numpy as np
from piper_sim.contracts import ARMS, FEATURE_NAMES, LOWER, UPPER, Observation


class ArmLayout:
    def __init__(self, arms):
        self.arms = tuple(arms)
        if self.arms not in (ARMS, ('right',)):
            raise ValueError('Supported simulation layouts: dual or right')
        self.names = tuple(f'{side}.{name}' for side in self.arms
                           for name in ('joint1','joint2','joint3','joint4','joint5','joint6','gripper_width'))
        self.legacy_ids = [FEATURE_NAMES.index(name) for name in self.names]
        self.lower, self.upper = LOWER[self.legacy_ids], UPPER[self.legacy_ids]

    def vector(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (len(self.names),) or not np.isfinite(value).all():
            raise ValueError('Expected finite values for '+str(self.names))
        return value.copy()

    def observation(self, position, timestamp, sim_time, images, image_timestamps, extras):
        if self.arms == ARMS:
            return Observation(position,timestamp,sim_time,images,image_timestamps,extras)
        return SimpleNamespace(position=self.vector(position),timestamp=timestamp,sim_time=sim_time,
                               images=images,image_timestamps=image_timestamps,extras=extras)

    def command(self, position, timestamp, sequence):
        return SimpleNamespace(position=self.vector(position),timestamp=timestamp,sequence=sequence,
                               arms=self.arms)


def right_model(xml):
    """Remove the absent robot with MuJoCo's existing model-editing API."""
    import mujoco
    import hashlib
    from pathlib import Path
    from piper_sim.config import ROOT
    key=hashlib.sha256((mujoco.__version__+Path(__file__).read_text()+xml).encode()).hexdigest()
    cache=ROOT/'outputs/right_arm_model_cache'/('right_'+key+'.xml')
    if cache.exists():
        return mujoco.MjModel.from_xml_path(str(cache))
    spec=mujoco.MjSpec.from_string(xml)
    # MuJoCo removes referencing joints/actuators/tendons along with the subtree.
    spec.delete(spec.body('left/base_link'))
    # Current RGB confirms the empty installation plate is still on the table.
    spec.body('left/mount_plate').name='station/vacant_mount_plate'
    model=spec.compile()
    cache.parent.mkdir(parents=True,exist_ok=True)
    import xml.etree.ElementTree as ET
    edited=ET.fromstring(spec.to_xml())
    # Preserve explicit inertial policy for the existing static visual meshes.
    edited.find('compiler').attrib.update(ET.fromstring(xml).find('compiler').attrib)
    cache.write_text(ET.tostring(edited,encoding='unicode'))
    return mujoco.MjModel.from_xml_path(str(cache))


def save_recorded_model(model, path):
    """Keep the source scene's explicit inertia policy in MuJoCo's XML export."""
    import mujoco
    import xml.etree.ElementTree as ET
    mujoco.mj_saveLastXML(str(path),model)
    tree=ET.parse(path)
    tree.getroot().find('compiler').set('inertiafromgeom','false')
    tree.write(path,encoding='unicode')

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import agent_challenge


class FrameContractTests(unittest.TestCase):
    def test_shift_is_difference_and_rotation_preserves_vector_length(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            trial=root/'self_evolution'/'source'; trial.mkdir(parents=True)
            (trial/'task.yaml').write_text('strip_yaw_deg: -90\n')
            with patch.object(agent_challenge,'DATA',root):
                evidence=agent_challenge.frame_contract('source',1.5,.5)
            np.testing.assert_allclose(evidence['same_displacement_in_socket']['vector'],[0,1],atol=1e-12)
            np.testing.assert_allclose(evidence['initial_holder_change']['vector'],[1,0])
            rotation=np.array(evidence['world_to_socket_xy_rotation'])
            np.testing.assert_allclose(rotation.T@rotation,np.eye(2),atol=1e-12)
            self.assertNotIn('recommended_offset',json.dumps(evidence))
            self.assertTrue(evidence['privileged'])

    def test_zero_shift_does_not_invent_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            trial=root/'self_evolution'/'source'; trial.mkdir(parents=True)
            (trial/'task.yaml').write_text('strip_yaw_deg: 37\n')
            with patch.object(agent_challenge,'DATA',root):
                evidence=agent_challenge.frame_contract('source',-2,-2)
            np.testing.assert_equal(evidence['same_displacement_in_socket']['vector'],[0,0])

if __name__=='__main__': unittest.main()

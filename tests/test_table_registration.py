import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_table_registration import plane_error,metrics


class TableRegistrationTests(unittest.TestCase):
    def test_plane_residual_uses_camera_transform_and_table_height(self):
        transform=np.eye(4);transform[2,2]=-1;transform[2,3]=.775
        error=plane_error(np.array([[0,0,.75],[0,0,.749]]),transform)
        np.testing.assert_allclose(error,[0,.001],atol=1e-15)
        self.assertAlmostEqual(metrics(error)['rmse_mm'],2**-.5)

"""Synthetic RGB and metric fixture. All dimensions and tolerances are test values."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import relative_pose as rp


def pose(angles, xyz):
    t = np.eye(4)
    for axis, angle in enumerate(angles):
        vector = np.eye(3)[axis] * np.deg2rad(angle)
        t[:3, :3] = cv2.Rodrigues(vector)[0] @ t[:3, :3]
    t[:3, 3] = xyz
    return t


def projected(ct, side, k, distortion):
    local = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * side / 2
    points, _ = cv2.projectPoints(local, cv2.Rodrigues(ct[:3, :3])[0], ct[:3, 3], k, distortion)
    return points.reshape(4, 2)


def render_tag(image, dictionary, tag_id, corners):
    n = 192
    if hasattr(cv2.aruco, 'generateImageMarker'):
        tag = cv2.aruco.generateImageMarker(dictionary, tag_id, n)
    else:
        tag = cv2.aruco.drawMarker(dictionary, tag_id, n)
    source = np.float32([[0, 0], [n-1, 0], [n-1, n-1], [0, n-1]])
    h = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    warped = cv2.warpPerspective(tag, h, image.shape[::-1], flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return np.minimum(image, warped)


def fixture():
    k = np.array([[860., 0, 340], [0, 810., 230], [0, 0, 1]])
    d = np.array([-.09, .01, .001, -.002, 0.])
    wc = pose([3, -7, 33], [.12, -.18, .33])
    wg = pose([26, -13, 19], [.58, .05, .77])
    offsets = {'socket': pose([11, 4, -14], [.023, -.004, .017]),
               'plug': pose([-4, -5, 9], [-.012, .013, .018])}
    sizes = {'socket': .079, 'plug': .069}
    markers = {r: {'id': i+3, 'side_m': sizes[r], 'T_tag_task': offsets[r].tolist()}
               for i, r in enumerate(rp.ROLES)}
    cal = {'domain': 'synthetic', 'geometry_ref': 'synthetic test geometry', 'calibration_version': 'test-v1',
           'K': k.tolist(), 'distortion': d.tolist(), 'image_size': [680, 480], 'T_WC': wc.tolist(),
           'dictionary': 'DICT_4X4_50', 'markers': markers, 'max_reprojection_px': 1.0,
           'pin_tips_P_m': [[-.006, 0., .007], [.006, 0., .007]], 'insertion_axis_P': [0, 0, 1],
           'task_tolerance': {'max_tip_error_m': .001, 'max_axis_error_deg': 1., 'source': 'synthetic fit'},
           'm0': {'calibration_version': 'test-v1', 'reference_kind': 'synthetic_oracle', 'cases': {}}}
    cs_tag = pose([151, -22, 4], [-.065, -.015, .47])
    cp_tag = pose([151, 26, -7], [.068, .018, .43])
    moves = {'nominal': np.eye(4), 'translation_1': pose([0, 0, 0], [.012, 0, 0]),
             'translation_2': pose([0, 0, 0], [0, .014, 0]),
             'rotation': pose([0, 8, 0], [0, 0, 0]), 'reseat': np.eye(4)}
    for case, motion in moves.items():
        ct = {'socket': cs_tag, 'plug': cp_tag @ motion}
        corners = {r: projected(ct[r], sizes[r], k, d).tolist() for r in rp.ROLES}
        ws = wc @ ct['socket'] @ offsets['socket']
        gp = np.linalg.inv(wg) @ wc @ ct['plug'] @ offsets['plug']
        cal['m0']['cases'][case] = {'T_WG': wg.tolist(), 'corners_px': corners,
                                    'capture_ref': 'render/' + case, 'reference_ref': 'oracle/' + case,
                                    'reference_T_WS': ws.tolist(), 'reference_T_GP': gp.tolist()}
    nominal = cal['m0']['cases']['nominal']
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((480, 680), 255, np.uint8)
    for role in rp.ROLES:
        image = render_tag(image, dictionary, markers[role]['id'], np.array(nominal['corners_px'][role]))
    return np.repeat(image[..., None], 3, axis=2), cal, wg


class RelativePoseTests(unittest.TestCase):
    def setUp(self):
        self.image, self.cal, self.wg = fixture()
        self.nominal = self.cal['m0']['cases']['nominal']

    def measure(self):
        return rp.measure_rgb(self.image, self.cal, self.wg, 12.5)

    def test_raster_detection_and_metric_frame_composition(self):
        result = self.measure()
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['source'], 'synthetic_fiducial_assisted')
        self.assertEqual(result['timestamp_s'], 12.5)
        self.assertEqual(result['calibration_version'], 'test-v1')
        for key in ('T_WS', 'T_GP'):
            expected, measured = np.array(self.nominal['reference_' + key]), np.array(result[key])
            np.testing.assert_allclose(measured[:3, 3], expected[:3, 3], atol=.003)
            np.testing.assert_allclose(measured[:3, :3], expected[:3, :3], atol=.025)
        for row in result['quality']['m0_cases'].values():
            self.assertLess(row['max_tip_error_m'], 1e-8)
            self.assertLess(row['max_axis_error_deg'], 1e-6)
        json.dumps(result, allow_nan=False)

    def test_exact_corners_recover_both_poses_without_extrinsic_cancellation(self):
        with patch.object(rp, '_detect', return_value=self.nominal['corners_px']):
            result = self.measure()
        self.assertTrue(result['valid'], result)
        for key in ('T_WS', 'T_GP'):
            np.testing.assert_allclose(result[key], self.nominal['reference_' + key], atol=1e-9)
        self.cal['T_WC'][0][3] += .01
        self.assertIn('M0 exceeds task tolerance', self.measure()['missing'][0])

    def test_two_plausible_planar_poses_refused(self):
        self.cal['max_reprojection_px'] = 100
        self.assertIn('ambiguous', self.measure()['missing'][0])

    def test_unqualified_camera_view_refused_even_at_low_reprojection_error(self):
        corners = {role: np.array(value) + [10, 0] for role, value in self.nominal['corners_px'].items()}
        with patch.object(rp, '_detect', return_value=corners):
            result = self.measure()
        self.assertIn('outside M0 viewing envelope', result['missing'][0])
        self.assertNotIn('T_WS', result)

    def test_missing_and_duplicate_tags_fail_without_pose(self):
        self.image.fill(255)
        self.assertIn('detected 0', self.measure()['missing'][0])
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        image = self.image[..., 0].copy()
        for x in (40, 370):
            corners = np.array([[x, 70], [x+90, 70], [x+90, 160], [x, 160]])
            image = render_tag(image, dictionary, self.cal['markers']['socket']['id'], corners)
        self.image = np.repeat(image[..., None], 3, axis=2)
        result = self.measure()
        self.assertIn('detected 2', result['missing'][0])
        self.assertNotIn('T_WS', result)

    def test_unqualified_geometry_reports_missing_or_rejected_inputs(self):
        absent = rp.measure_rgb(None, None, None, None)
        for field in ('frame', 'T_WG', 'T_WC', 'task_tolerance', 'm0'):
            self.assertIn(field, absent['missing'])
        # Synthetic qualification cannot be used as physical M0.
        self.cal['domain'] = 'real'
        self.assertFalse(self.measure()['valid'])
        self.cal['domain'] = 'synthetic'
        self.cal['m0']['cases']['rotation']['reference_T_GP'][0][3] += .02
        result = self.measure()
        self.assertIn('M0 exceeds task tolerance', result['missing'][0])
        self.assertNotIn('T_GP', result)

    def test_calibration_version_and_repeated_views_fail_qualification(self):
        self.cal['m0']['calibration_version'] = 'other'
        self.assertFalse(self.measure()['valid'])
        self.cal['m0']['calibration_version'] = 'test-v1'
        duplicate = copy.deepcopy(self.cal['m0']['cases']['translation_1'])
        duplicate['capture_ref'] = 'independent frame with the same translation'
        self.cal['m0']['cases']['translation_2'] = duplicate
        self.assertIn('two resolved translation', self.measure()['missing'][0])

    def test_wrong_image_shape_malformed_transform_and_missing_identity_refused(self):
        self.image = self.image[2:]
        self.assertIn('calibrated image_size', self.measure()['missing'][0])
        self.image, self.cal, self.wg = fixture()
        self.wg[1, 1] = 5
        self.assertIn('rigid', self.measure()['missing'][0])
        self.image, self.cal, self.wg = fixture()
        del self.cal['m0']['cases']['reseat']
        self.assertIn('reseat', self.measure()['missing'][0])

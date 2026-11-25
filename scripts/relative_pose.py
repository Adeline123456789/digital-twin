"""Measure fixed-view fiducials at a stopped task boundary. T_AB maps B to A, in metres.

A local calibration supplies the pinhole K, OpenCV distortion coefficients, calibrated
image_size [width,height], camera-to-world T_WC, dictionary name (e.g. DICT_APRILTAG_36h11),
and markers.socket / markers.plug. Each marker has id, side_m (outer black square)
and T_tag_task (task coordinates in the tag frame). geometry_ref identifies these
measurements. The caller pairs the RGB exposure and right-tool FK on the same clock.

M0 re-solves reviewed corner observations with this exact geometry. Each case supplies
capture_ref, reference_ref, T_WG, corners_px by marker role (decoded top-left clockwise),
and independent reference_T_WS / reference_T_GP. The metric pin_tips_P_m and unit
insertion_axis_P describe the plug. task_tolerance gives max_tip_error_m,
max_axis_error_deg and a source for the physical allowance. No camera defaults.
"""
import numpy as np


ROLES = ('socket', 'plug')
CASES = ('nominal', 'translation_1', 'translation_2', 'rotation', 'reseat')
FIELDS = ('domain', 'calibration_version', 'geometry_ref', 'image_size', 'K', 'distortion',
          'T_WC', 'dictionary', 'markers', 'max_reprojection_px', 'pin_tips_P_m',
          'insertion_axis_P', 'task_tolerance', 'm0')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _label(value, name):
    _require(isinstance(value, str) and value.strip(), name)
    return value


def _array(value, name, shape=None):
    arr = np.asarray(value, dtype=float)
    _require(np.isfinite(arr).all() and (shape is None or arr.shape == shape), name)
    return arr


def _rigid(value, name):
    t = _array(value, name, (4, 4))
    rot = t[:3, :3]
    _require(np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
             and np.allclose(rot.T @ rot, np.eye(3), atol=1e-7, rtol=0)
             and np.linalg.det(rot) > 0, name + ' must be a rigid transform')
    return t


def _positive(value, name):
    _require(not isinstance(value, (bool, np.bool_)) and np.isscalar(value), name)
    value = float(value)
    _require(np.isfinite(value) and value > 0, name)
    return value


def _detect(cv2, rgb, dictionary, ids):
    aruco = cv2.aruco
    _require(dictionary.startswith('DICT_') and hasattr(aruco, dictionary),
             'supported fiducial dictionary')
    codebook = aruco.getPredefinedDictionary(getattr(aruco, dictionary))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if hasattr(aruco, 'ArucoDetector'):
        corners, found, _ = aruco.ArucoDetector(codebook).detectMarkers(gray)
    else:
        corners, found, _ = aruco.detectMarkers(gray, codebook)
    found = np.array([], dtype=int) if found is None else found.ravel()
    result = {}
    for role, tag_id in ids.items():
        matches = np.flatnonzero(found == tag_id)
        _require(len(matches) == 1, f'exactly one {role} tag {tag_id} (detected {len(matches)})')
        result[role] = corners[matches[0]].reshape(4, 2)
    return result


def _solve(cv2, corners, side, intrinsics, distortion, limit, image_size, role):
    pixels = _array(corners, role + ' ordered corners', (4, 2))
    _require(((pixels >= 0) & (pixels < image_size)).all(), role + ' corners outside calibrated image')
    contour = pixels.astype(np.float32)
    _require(cv2.isContourConvex(contour) and cv2.contourArea(contour, oriented=True) > 0,
             role + ' requires clockwise decoded corners')
    points = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * (side / 2)
    count, rotations, translations, _ = cv2.solvePnPGeneric(
        points, pixels, intrinsics, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    _require(count == 2, role + ' unresolved planar branches')
    plausible = []
    for rotation, translation in zip(rotations, translations):
        _require(np.isfinite(rotation).all() and np.isfinite(translation).all(),
                 role + ' degenerate planar pose')
        t = np.eye(4)
        t[:3, :3] = cv2.Rodrigues(rotation)[0]
        t[:3, 3] = translation.ravel()
        camera_points = points @ t[:3, :3].T + t[:3, 3]
        if min(camera_points[:, 2]) <= 0 or t[:3, 2] @ t[:3, 3] >= 0:
            continue
        projected, _ = cv2.projectPoints(points, rotation, translation, intrinsics, distortion)
        delta = projected.reshape(4, 2) - pixels
        rms = float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))))
        if rms <= limit:
            plausible.append((t, rms))
    _require(plausible, role + ' has no front-facing pose within reprojection bound')
    _require(len(plausible) == 1, role + ' planar pose ambiguous at calibrated pixel error')
    return plausible[0]


def task_error(reference, measured, tips, axis):
    def endpoints(t):
        return tips @ t[:3, :3].T + t[:3, 3]
    tip_error = np.linalg.norm(endpoints(reference) - endpoints(measured), axis=1).max()
    a, b = reference[:3, :3] @ axis, measured[:3, :3] @ axis
    axis_error = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))
    return float(tip_error), float(axis_error)


def _qualification(cal, poses, current_corners, pixel_limit):
    m0 = cal['m0']
    kind = {'real': 'independent_physical_fixture', 'synthetic': 'synthetic_oracle'}[cal['domain']]
    _require(m0['reference_kind'] == kind and m0['calibration_version'] == cal['calibration_version'],
             'matching M0 source and calibration_version')
    tips = _array(cal['pin_tips_P_m'], 'pin_tips_P_m')
    _require(tips.ndim == 2 and tips.shape[1] == 3 and len(np.unique(tips, axis=0)) >= 2,
             'two distinct metric pin tips')
    axis = _array(cal['insertion_axis_P'], 'insertion_axis_P', (3,))
    _require(abs(np.linalg.norm(axis) - 1) < 1e-7, 'unit insertion_axis_P')
    tolerance = cal['task_tolerance']
    _label(tolerance['source'], 'assembly allowance or recovery range source')
    pos_limit = _positive(tolerance['max_tip_error_m'], 'max_tip_error_m')
    ang_limit = _positive(tolerance['max_axis_error_deg'], 'max_axis_error_deg')
    records, references, captures = {}, {}, set()
    for case in CASES:
        sample = m0['cases'][case]
        capture = _label(sample['capture_ref'], case + ' RGB capture source')
        reference = _label(sample['reference_ref'], case + ' independent fixture source')
        _require(reference != capture and capture not in captures, case + ' independent observations')
        captures.add(capture)
        wg = _rigid(sample['T_WG'], case + ' T_WG')
        ws, gp, _ = poses(sample['corners_px'], wg)
        ref_ws = _rigid(sample['reference_T_WS'], case + ' independent T_WS')
        ref_gp = _rigid(sample['reference_T_GP'], case + ' independent T_GP')
        ref_sp = np.linalg.inv(ref_ws) @ wg @ ref_gp
        sp = np.linalg.inv(ws) @ wg @ gp
        # Evaluate common-camera cancellation as well as each reported pose at the tips.
        errors = [task_error(a, b, tips, axis) for a, b in
                  ((ref_sp, sp), (ref_gp, gp), (ref_ws @ ref_sp, ws @ ref_sp))]
        peak_tip, peak_axis = np.max(errors, axis=0)
        _require(peak_tip <= pos_limit and peak_axis <= ang_limit,
                 f'{case} M0 exceeds task tolerance (tip {peak_tip:g} m, axis {peak_axis:g} deg)')
        records[case] = {'max_tip_error_m': float(peak_tip), 'max_axis_error_deg': float(peak_axis)}
        references[case] = ref_sp
    nominal = references['nominal']
    shifts = [references[k][:3, 3] - nominal[:3, 3] for k in ('translation_1', 'translation_2')]
    _require(np.linalg.matrix_rank(shifts, tol=pos_limit) == 2, 'M0 needs two resolved translation directions')
    rotation = nominal[:3, :3].T @ references['rotation'][:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    _require(angle > ang_limit, 'M0 needs a resolved independent rotation')
    returned_tip, returned_axis = task_error(nominal, references['reseat'], tips, axis)
    _require(returned_tip <= pos_limit and returned_axis <= ang_limit, 'M0 independent return to nominal')
    # An excellent reprojection fit can still extrapolate to an untested view or scale.
    # Four-corner RMS <= limit permits an individual corner residual up to 2 * limit.
    for role in ROLES:
        sampled = np.array([m0['cases'][case]['corners_px'][role] for case in CASES])
        now = current_corners[role]
        _require(np.all(now >= sampled.min(axis=0) - 2 * pixel_limit)
                 and np.all(now <= sampled.max(axis=0) + 2 * pixel_limit),
                 role + ' tag corners outside M0 viewing envelope')
    return records


def measure_rgb(frame, calibration, T_WG, timestamp_s):
    """Return a qualified pair of poses or explicit missing evidence. Freshness is checked by the caller."""
    cal = calibration if isinstance(calibration, dict) else {}
    missing = [k for k in FIELDS if cal.get(k) is None]
    missing += [k for k, v in (('frame', frame), ('T_WG', T_WG), ('timestamp_s', timestamp_s)) if v is None]
    if missing:
        return {'valid': False, 'reason': 'missing RGB measurement inputs', 'missing': missing}
    try:
        import cv2
    except ImportError:
        return {'valid': False, 'reason': 'OpenCV unavailable', 'missing': ['OpenCV with ArUco and IPPE']}
    try:
        _require(cal['domain'] in ('real', 'synthetic'), 'real or synthetic calibration domain')
        version = _label(cal['calibration_version'], 'calibration_version')
        _label(cal['geometry_ref'], 'source for measured geometry')
        stamp = float(_array(timestamp_s, 'RGB exposure timestamp', ()))
        wg = _rigid(T_WG, 'T_WG')
        wc = _rigid(cal['T_WC'], 'fixed-camera T_WC')
        rgb = np.asarray(frame)
        size = _array(cal['image_size'], 'image_size', (2,))
        _require(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8 and rgb.size
                 and np.array_equal(size, rgb.shape[1::-1]), 'uint8 RGB frame at calibrated image_size')
        k = _array(cal['K'], 'K', (3, 3))
        _require(k[0, 0] > 0 and k[1, 1] > 0 and k[0, 1] == k[1, 0] == 0
                 and np.array_equal(k[2], [0, 0, 1]), 'pinhole RGB intrinsics')
        dist = _array(cal['distortion'], 'distortion')
        _require(dist.ndim == 1 and dist.size in (4, 5, 8, 12, 14), 'OpenCV pinhole distortion coefficients')
        limit = _positive(cal['max_reprojection_px'], 'calibrated corner reprojection bound')
        markers = cal['markers']
        ids = {role: markers[role]['id'] for role in ROLES}
        _require(all(type(tag) is int and tag >= 0 for tag in ids.values()) and len(set(ids.values())) == 2,
                 'distinct socket and plug marker IDs')
        sizes = {r: _positive(markers[r]['side_m'], r + ' tag side_m') for r in ROLES}
        offsets = {r: _rigid(markers[r]['T_tag_task'], r + ' T_tag_task') for r in ROLES}

        def poses(corners, tool):
            world, residuals = {}, {}
            for role in ROLES:
                ct, rms = _solve(cv2, corners[role], sizes[role], k, dist, limit, size, role)
                world[role] = wc @ ct @ offsets[role]
                residuals[role] = rms
            return world['socket'], np.linalg.inv(tool) @ world['plug'], residuals

        corners = _detect(cv2, rgb, _label(cal['dictionary'], 'dictionary'), ids)
        ws, gp, residuals = poses(corners, wg)
        cases = _qualification(cal, poses, corners, limit)
        source = 'fiducial_assisted' if cal['domain'] == 'real' else 'synthetic_fiducial_assisted'
        return {'valid': True, 'T_WS': ws.tolist(), 'T_GP': gp.tolist(), 'source': source,
                'timestamp_s': stamp, 'calibration_version': version,
                'quality': {'reprojection_rms_px': residuals, 'm0_cases': cases,
                            'task_tolerance': {key: cal['task_tolerance'][key] for key in
                                               ('max_tip_error_m', 'max_axis_error_deg', 'source')}}}
    except (AttributeError, KeyError, ValueError, TypeError, OverflowError, np.linalg.LinAlgError, cv2.error) as exc:
        return {'valid': False, 'reason': 'RGB observation or qualification rejected', 'missing': [str(exc)]}

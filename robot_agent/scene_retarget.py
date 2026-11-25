"""Stage 0: RGB-only initial scene alignment for fixed tabletop strip and plug holder.

Computes differential world XY/yaw corrections on the known tabletop/fixture planes
from an unobstructed external RGB frame (simulated overhead or real D455F RGB),
never using depth maps or simulator privileged object coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


# World Z coordinates (table top = +0.025 m):
#   Strip blue socket face = 0.025 + 0.0283 = 0.0533 m
#   Yellow plug visible housing centroid on holder = ~0.0720 m
STRIP_FACE_Z_M = 0.0533
HOLDER_TOP_Z_M = 0.0330
PLUG_HOUSING_Z_M = 0.0720


@dataclass(frozen=True)
class SceneAlignmentResult:
    valid: bool
    method: str
    confidence: float
    strip_delta_xy_m: tuple[float, float]
    strip_delta_yaw_deg: float
    holder_delta_xy_m: tuple[float, float]
    holder_delta_yaw_deg: float
    outlet3_px: tuple[float, float] | None
    plug_px: tuple[float, float] | None
    details: dict


def pixel_to_world_at_z(
    uv: tuple[float, float] | np.ndarray,
    z_world_m: float,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    distortion: np.ndarray | None = None,
) -> np.ndarray:
    """Intersect an undistorted camera ray with horizontal world plane Z = z_world_m."""
    K_mat = np.asarray(K, dtype=np.float64).reshape(3, 3)
    T_wc = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(uv, dtype=np.float64).reshape(1, 1, 2)
    dist = np.zeros(5, dtype=np.float64) if distortion is None else np.asarray(distortion, dtype=np.float64)
    undist = cv2.undistortPoints(pts, K_mat, dist)[0, 0]
    ray_cam = np.array([undist[0], undist[1], 1.0], dtype=np.float64)
    ray_world = T_wc[:3, :3] @ ray_cam
    cam_world = T_wc[:3, 3]
    if abs(ray_world[2]) < 1e-6:
        raise ValueError("Camera ray nearly parallel to tabletop plane")
    scale = (float(z_world_m) - cam_world[2]) / ray_world[2]
    if scale <= 0:
        raise ValueError("Plane intersection behind camera")
    return cam_world + scale * ray_world


def _four_panel_features(blue: np.ndarray) -> dict | None:
    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area >= 60 and 10 <= bw <= 50 and 10 <= bh <= 50 and 0.45 <= bw / max(1, bh) <= 2.2:
            m = cv2.moments(c)
            if m["m00"] > 0:
                panels.append((m["m10"] / m["m00"], m["m01"] / m["m00"], bw, bh))

    panels.sort(key=lambda item: item[0])
    if len(panels) >= 4:
        # Select best 4 collinear equally spaced blue modules
        best = None
        from itertools import combinations
        for grp in combinations(panels, 4):
            pts = np.array([[p[0], p[1]] for p in grp], dtype=np.float64)
            widths = np.array([p[2] for p in grp], dtype=np.float64)
            gaps = np.diff(pts[:, 0])
            if (widths.max() / max(1e-6, widths.min()) < 1.55
                    and np.ptp(pts[:, 1]) < widths.mean() * 0.45
                    and np.all((gaps > widths.mean() * 0.75) & (gaps < widths.mean() * 2.1))):
                score = float(np.std(gaps))
                if best is None or score < best[0]:
                    best = (score, pts)
        if best is not None:
            pts = best[1]
            # Fit line across the 4 module centers for subpixel image yaw
            vx, vy, _, _ = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            if vx < 0:
                vx, vy = -vx, -vy
            yaw_img_deg = float(np.rad2deg(np.arctan2(vy, vx)))
            return {
                "panels_px": [tuple(map(float, p)) for p in pts],
                "outlet3_px": (float(pts[3, 0]), float(pts[3, 1])),
                "strip_center_px": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "image_yaw_deg": yaw_img_deg,
                "confidence": 0.95,
            }
    return None


def detect_strip_features_rgb(rgb: np.ndarray) -> dict | None:
    """Detect the white power strip and its 4 blue panels [USB, AC1, AC2, AC3] in RGB."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB image")
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # ROI covering tabletop center where the power strip sits (excluding grey plug holder on the right >= 0.63*w)
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[int(0.32 * h):int(0.74 * h), int(0.24 * w):int(0.63 * w)] = 255

    # Try saturated panel interiors first under a blue white-balance cast.
    # The original mask still resolves desaturated panels in other lighting.
    for saturation in (78, 42):
        blue = cv2.inRange(hsv, (92, saturation, 28), (132, 255, 255)) & roi
        features = _four_panel_features(blue)
        if features is not None:
            return features

    # Fallback: merged blue band + white strip body rectangle when glare merges blue panels
    merged = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((5, 15), dtype=np.uint8))
    mcnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in mcnts if cv2.contourArea(c) >= 400]
    if not valid:
        return None
    band = max(valid, key=cv2.contourArea)
    rect = cv2.minAreaRect(band)
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    # Outlet 3 center is +1.5 * pitch (~ +0.375 of 4-module span) to the right of band center
    rad = np.deg2rad(angle)
    offset = 0.365 * rw
    outlet3 = (float(cx + offset * np.cos(rad)), float(cy + offset * np.sin(rad)))
    return {
        "panels_px": [],
        "outlet3_px": outlet3,
        "strip_center_px": (float(cx), float(cy)),
        "image_yaw_deg": float(angle),
        "confidence": 0.72,
    }


def detect_holder_plug_rgb(rgb: np.ndarray) -> dict | None:
    """Detect the yellow plug housing and grey holder orientation in external RGB."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB image")
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[int(0.32 * h):int(0.72 * h), int(0.62 * w):int(0.97 * w)] = 255

    yellow = cv2.inRange(hsv, (15, 75, 45), (43, 255, 255)) & roi
    cnts, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) >= 60]
    if not valid:
        return None
    plug_cnt = max(valid, key=cv2.contourArea)
    m = cv2.moments(plug_cnt)
    if m["m00"] <= 0:
        return None
    px, py = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])

    # Estimate holder top-edge orientation in a local window above the folded right gripper
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    y0 = max(0, int(py - 32))
    y1 = min(h, int(py + 20))
    x0 = max(0, int(px - 70))
    x1 = min(w, int(px + 65))
    patch_gray = gray[y0:y1, x0:x1]
    holder_mask = (patch_gray < 150).astype(np.uint8) * 255
    hcnts, _ = cv2.findContours(holder_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yaw_img_deg = 0.0
    if hcnts:
        hc = max(hcnts, key=cv2.contourArea)
        if cv2.contourArea(hc) >= 300:
            (_, _), (rw, rh), ang = cv2.minAreaRect(hc)
            if rw < rh:
                ang += 90.0
            if ang > 45.0:
                ang -= 90.0
            elif ang < -45.0:
                ang += 90.0
            yaw_img_deg = float(ang)

    return {
        "plug_center_px": (px, py),
        "image_yaw_deg": yaw_img_deg,
        "confidence": 0.90,
    }


def _ecc_patch_delta(
    ref_rgb: np.ndarray,
    cur_rgb: np.ndarray,
    center_px: tuple[float, float],
    half_w: int,
    half_h: int,
    query_px: tuple[float, float] | None = None,
    search_pad: int = 28,
) -> tuple[float, np.ndarray, float] | None:
    """Subpixel rigid registration around a reference object patch, evaluated at query_px."""
    h, w = ref_rgb.shape[:2]
    cx, cy = int(round(center_px[0])), int(round(center_px[1]))
    x0, x1 = max(0, cx - half_w), min(w, cx + half_w)
    y0, y1 = max(0, cy - half_h), min(h, cy + half_h)
    if x1 - x0 < 24 or y1 - y0 < 18:
        return None
    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cur_gray = cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tmpl = ref_gray[y0:y1, x0:x1]

    sx0, sx1 = max(0, x0 - search_pad), min(w, x1 + search_pad)
    sy0, sy1 = max(0, y0 - search_pad), min(h, y1 + search_pad)
    roi = cur_gray[sy0:sy1, sx0:sx1]
    if roi.shape[0] <= tmpl.shape[0] or roi.shape[1] <= tmpl.shape[1]:
        return None
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.65:
        return None

    init_x = float(sx0 + max_loc[0])
    init_y = float(sy0 + max_loc[1])
    warp = np.array([[1.0, 0.0, init_x], [0.0, 1.0, init_y]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5)
    try:
        cc, warp = cv2.findTransformECC(tmpl, cur_gray, warp, cv2.MOTION_EUCLIDEAN, criteria)
    except cv2.error:
        cc = float(max_val)
    ref_q = np.asarray(query_px if query_px is not None else [(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
    local_q = ref_q - np.array([x0, y0], dtype=np.float64)
    cur_q = warp[:, :2].astype(np.float64) @ local_q + warp[:, 2].astype(np.float64)
    dtheta_deg = float(np.rad2deg(np.arctan2(warp[1, 0], warp[0, 0])))
    return float(cc), cur_q - ref_q, dtheta_deg


def align_scene_from_external_rgb(
    current_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    distortion: np.ndarray | None = None,
    max_shift_m: float = 0.045,
    max_yaw_deg: float = 12.0,
) -> SceneAlignmentResult:
    """Estimate differential strip & holder world XY/yaw offsets from external RGB.

    Anchored to a verified reference external RGB image so static camera extrinsic
    bias cancels to first order while shifts/rotations of the strip or holder are
    recovered accurately.
    """
    ref_strip = detect_strip_features_rgb(reference_rgb)
    cur_strip = detect_strip_features_rgb(current_rgb)
    ref_holder = detect_holder_plug_rgb(reference_rgb)
    cur_holder = detect_holder_plug_rgb(current_rgb)

    if ref_strip is None or cur_strip is None or ref_holder is None or cur_holder is None:
        return SceneAlignmentResult(
            valid=False,
            method="UNRESOLVED_RGB",
            confidence=0.0,
            strip_delta_xy_m=(0.0, 0.0),
            strip_delta_yaw_deg=0.0,
            holder_delta_xy_m=(0.0, 0.0),
            holder_delta_yaw_deg=0.0,
            outlet3_px=cur_strip["outlet3_px"] if cur_strip else None,
            plug_px=cur_holder["plug_center_px"] if cur_holder else None,
            details={"reason": "Missing strip or yellow plug features in external RGB"},
        )

    # 1. Semantic ray-plane differential estimate
    s_ref_w = pixel_to_world_at_z(ref_strip["outlet3_px"], STRIP_FACE_Z_M, K, T_world_cam, distortion)
    s_cur_w = pixel_to_world_at_z(cur_strip["outlet3_px"], STRIP_FACE_Z_M, K, T_world_cam, distortion)
    sem_strip_dxy = s_cur_w[:2] - s_ref_w[:2]
    # Overhead image +u corresponds to world -Y, +v corresponds to world -X -> clockwise image rotation = -world yaw
    sem_strip_dyaw = -(cur_strip["image_yaw_deg"] - ref_strip["image_yaw_deg"])

    h_ref_w = pixel_to_world_at_z(ref_holder["plug_center_px"], PLUG_HOUSING_Z_M, K, T_world_cam, distortion)
    h_cur_w = pixel_to_world_at_z(cur_holder["plug_center_px"], PLUG_HOUSING_Z_M, K, T_world_cam, distortion)
    sem_holder_dxy = h_cur_w[:2] - h_ref_w[:2]
    sem_holder_dyaw = -(cur_holder["image_yaw_deg"] - ref_holder["image_yaw_deg"])

    # 2. Subpixel ECC cross-check evaluated directly at outlet3_px and plug_center_px!
    ecc_s = _ecc_patch_delta(
        reference_rgb, current_rgb, ref_strip["strip_center_px"], half_w=62, half_h=24, query_px=ref_strip["outlet3_px"]
    )
    ecc_h = _ecc_patch_delta(
        reference_rgb,
        current_rgb,
        (ref_holder["plug_center_px"][0], ref_holder["plug_center_px"][1] - 6.0),
        half_w=52,
        half_h=18,
        query_px=ref_holder["plug_center_px"],
    )

    method = "semantic_rgb_differential"
    strip_dxy = sem_strip_dxy.copy()
    strip_dyaw = float(sem_strip_dyaw)
    if ecc_s is not None and ecc_s[0] >= 0.88:
        cc_s, dpx_s, dtheta_s = ecc_s
        c0 = np.asarray(ref_strip["outlet3_px"], dtype=np.float64)
        w0 = pixel_to_world_at_z(c0, STRIP_FACE_Z_M, K, T_world_cam, distortion)
        w1 = pixel_to_world_at_z(c0 + dpx_s, STRIP_FACE_Z_M, K, T_world_cam, distortion)
        ecc_dxy = w1[:2] - w0[:2]
        if cc_s >= 0.92:
            strip_dxy = ecc_dxy
            strip_dyaw = float(-dtheta_s)
            method = "ecc_subpixel_rgb"
        elif np.linalg.norm(ecc_dxy - sem_strip_dxy) <= 0.008:
            strip_dxy = 0.65 * ecc_dxy + 0.35 * sem_strip_dxy
            strip_dyaw = 0.65 * (-dtheta_s) + 0.35 * sem_strip_dyaw
            method = "hybrid_ecc_semantic_rgb"

    holder_dxy = sem_holder_dxy.copy()
    holder_dyaw = float(sem_holder_dyaw)
    if ecc_h is not None and ecc_h[0] >= 0.88:
        cc_h, dpx_h, dtheta_h = ecc_h
        p0 = np.asarray(ref_holder["plug_center_px"], dtype=np.float64)
        w0 = pixel_to_world_at_z(p0, PLUG_HOUSING_Z_M, K, T_world_cam, distortion)
        w1 = pixel_to_world_at_z(p0 + dpx_h, PLUG_HOUSING_Z_M, K, T_world_cam, distortion)
        ecc_hdxy = w1[:2] - w0[:2]
        if cc_h >= 0.92:
            holder_dxy = ecc_hdxy
            holder_dyaw = float(-dtheta_h)
            method = "ecc_subpixel_rgb"
        elif np.linalg.norm(ecc_hdxy - sem_holder_dxy) <= 0.008:
            holder_dxy = 0.65 * ecc_hdxy + 0.35 * sem_holder_dxy
            holder_dyaw = 0.65 * (-dtheta_h) + 0.35 * sem_holder_dyaw
            method = "hybrid_ecc_semantic_rgb"

    strip_dxy = np.clip(strip_dxy, -max_shift_m, max_shift_m)
    holder_dxy = np.clip(holder_dxy, -max_shift_m, max_shift_m)
    strip_dyaw = float(np.clip(strip_dyaw, -max_yaw_deg, max_yaw_deg))
    holder_dyaw = float(np.clip(holder_dyaw, -max_yaw_deg, max_yaw_deg))

    conf = float(min(cur_strip["confidence"], cur_holder["confidence"]))
    return SceneAlignmentResult(
        valid=True,
        method=method,
        confidence=conf,
        strip_delta_xy_m=(float(strip_dxy[0]), float(strip_dxy[1])),
        strip_delta_yaw_deg=strip_dyaw,
        holder_delta_xy_m=(float(holder_dxy[0]), float(holder_dxy[1])),
        holder_delta_yaw_deg=holder_dyaw,
        outlet3_px=cur_strip["outlet3_px"],
        plug_px=cur_holder["plug_center_px"],
        details={
            "semantic_strip_delta_mm": (sem_strip_dxy * 1000.0).tolist(),
            "semantic_holder_delta_mm": (sem_holder_dxy * 1000.0).tolist(),
            "ecc_strip_cc": None if ecc_s is None else round(ecc_s[0], 4),
            "ecc_holder_cc": None if ecc_h is None else round(ecc_h[0], 4),
        },
    )

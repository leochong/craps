import json
import os

import cv2
import numpy as np

from .pose import default_intrinsics

GREEN_FELT_LOWER = np.array([35, 40, 40], dtype=np.uint8)
GREEN_FELT_UPPER = np.array([85, 255, 255], dtype=np.uint8)


def felt_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_FELT_LOWER, GREEN_FELT_UPPER)
    return mask


def detect_plate_circle(bgr, min_radius=None, max_radius=None):
    h, w = bgr.shape[:2]
    if min_radius is None:
        min_radius = int(0.10 * min(h, w))
    if max_radius is None:
        max_radius = int(0.60 * min(h, w))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    finest = None
    for param2 in (40, 60, 80):
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=0.5 * max_radius,
            param1=100,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue
        circles = np.round(circles[0]).astype(int)
        for (x, y, r) in circles:
            if finest is None or r > finest[2]:
                finest = (x, y, r)

    if finest is None:
        return None
    x, y, r = finest
    return {"cx": x, "cy": y, "radius": r}


def plate_origin(bgr, scale_mm_per_pixel=1.0):
    circle = detect_plate_circle(bgr)
    if circle is None:
        return None
    circle["mask"] = felt_mask(bgr)
    circle["radius_mm"] = circle["radius"] * scale_mm_per_pixel
    return circle


def detect_plate_ellipse(bgr, min_area_frac=0.02):
    """Fit an ellipse to the green felt plate (robust to a tilted overhead view)."""
    h, w = bgr.shape[:2]
    mask = felt_mask(bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_frac * h * w or len(contour) < 5:
        return None
    (cx, cy), (major, minor), angle = cv2.fitEllipse(contour)
    return {
        "cx": float(cx),
        "cy": float(cy),
        "radius": float((major + minor) / 4.0),
        "major_px": float(major / 2.0),
        "minor_px": float(minor / 2.0),
        "angle": float(angle),
        "source": "felt_ellipse",
    }


def _plausible_circle(circle, width, height):
    r = float(circle.get("radius", 0.0))
    cx, cy = float(circle["cx"]), float(circle["cy"])
    if not (0.03 * min(width, height) <= r <= 0.55 * min(width, height)):
        return False
    if not (0.0 <= cx <= width and 0.0 <= cy <= height):
        return False
    return True


def _ellipse_conic(cx, cy, a, b, angle_deg):
    """3x3 image conic for an ellipse with semi-axes (a, b) and rotation angle."""
    t = np.deg2rad(angle_deg)
    ct, st = np.cos(t), np.sin(t)
    inv = np.diag([1.0 / (a * a), 1.0 / (b * b)])
    rot = np.array([[ct, st], [-st, ct]])
    q2 = rot.T @ inv @ rot
    cu = np.zeros((3, 3))
    cu[:2, :2] = q2
    cu[2, 2] = -1.0
    shift = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    return shift.T @ cu @ shift


def _sample_world_circle(radius_m, n=64):
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([radius_m * np.cos(phi), radius_m * np.sin(phi), np.zeros(n)], axis=1)


def _project(R, t, K, world_pts):
    cam = world_pts @ R.T + t
    z = np.where(np.abs(cam[:, 2]) < 1e-9, 1e-9, cam[:, 2])
    uv = (cam[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv


def _circle_conic_residual(R, t, K, conic, radius_m):
    uv = _project(R, t, K, _sample_world_circle(radius_m))
    homog = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    val = np.einsum("ij,jk,ik->i", homog, conic, homog)
    return float(np.mean(val ** 2))


def _solve_circle_scale(R, t_dir, K, conic, radius_m):
    best_s, best_r = 1.0, np.inf
    for s in np.geomspace(0.02, 8.0, 240):
        r = _circle_conic_residual(R, s * t_dir, K, conic, radius_m)
        if r < best_r:
            best_r, best_s = r, s
    lo, hi = best_s * 0.7, best_s * 1.4
    for _ in range(40):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _circle_conic_residual(R, m1 * t_dir, K, conic, radius_m) < \
                _circle_conic_residual(R, m2 * t_dir, K, conic, radius_m):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def _nelder_mead(fun, x0, step=0.1, iters=400):
    """Compact Nelder-Mead minimiser (no scipy dependency)."""
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.size
    pts = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] += step
        pts.append(p)
    pts = np.array(pts)
    vals = np.array([fun(p) for p in pts])
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = pts[order], vals[order]
        centroid = pts[:-1].mean(axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = fun(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - pts[-1])
            fe = fun(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (pts[-1] - centroid)
            fc = fun(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = pts[0] + 0.5 * (pts - pts[0])
                vals = np.array([fun(p) for p in pts])
    order = np.argsort(vals)
    return pts[order][0], float(vals[order][0])


def _plane_pose_cost(params, K, conic, radius_m):
    R, _ = cv2.Rodrigues(np.asarray(params[:3], dtype=np.float64))
    return _circle_conic_residual(R, np.asarray(params[3:], dtype=np.float64),
                                  K, conic, radius_m)


def _rotation_with_normal(normal):
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(normal, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    r1 = np.cross(ref, normal)
    r1 = r1 / np.linalg.norm(r1)
    r2 = np.cross(normal, r1)
    return np.stack([r1, r2, normal], axis=1)


def _circle_pose_inits(ellipse, conic, K, radius_m):
    t_dir = np.linalg.inv(K) @ np.array([ellipse["cx"], ellipse["cy"], 1.0])
    fronto = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    inits = [(fronto, _solve_circle_scale(fronto, t_dir, K, conic, radius_m) * t_dir)]
    cam_conic = K.T @ conic @ K
    _, vecs = np.linalg.eigh(cam_conic)
    normal = vecs[:, 0]
    normal = normal / np.linalg.norm(normal)
    if normal[2] > 0.0:
        normal = -normal
    R_n = _rotation_with_normal(normal)
    t_dir = -normal
    inits.append((R_n, _solve_circle_scale(R_n, t_dir, K, conic, radius_m) * t_dir))
    return inits


def pose_from_circle(ellipse, K, plate_radius_m, refine=True, iters=400):
    """Recover camera pose (R, t) of the floor plane from the felt ellipse.

    The plate is a known circle in the plane z=0; the image conic fixes the plane
    normal and the known radius fixes translation scale. The pose is polished by a
    Nelder-Mead search on the conic residual. In-plane rotation is a gauge freedom
    (only the normal and t are meaningful).
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    a = float(ellipse.get("major_px", ellipse.get("radius", 1.0)))
    b = float(ellipse.get("minor_px", ellipse.get("radius", 1.0)))
    conic = _ellipse_conic(ellipse["cx"], ellipse["cy"], a, b, ellipse.get("angle", 0.0))
    conic = conic / max(float(np.abs(conic).max()), 1e-30)

    best = None
    for R0, t0 in _circle_pose_inits(ellipse, conic, K, plate_radius_m):
        if refine:
            p0 = np.concatenate([cv2.Rodrigues(R0)[0].ravel(), t0])
            local = None
            for step in (0.15, 0.05, 0.02):
                p, v = _nelder_mead(lambda q: _plane_pose_cost(q, K, conic, plate_radius_m),
                                    p0, step=step, iters=iters)
                if local is None or v < local[1]:
                    local = (p, v)
            R, _ = cv2.Rodrigues(local[0][:3])
            t, resid = local[0][3:], local[1]
        else:
            R, t = R0, t0
            resid = _circle_conic_residual(R, t, K, conic, plate_radius_m)
        if t[2] <= 0.0:
            continue
        if best is None or resid < best["residual"]:
            best = {"R": R, "t": t, "normal": R[:, 2].copy(), "center_cam": t.copy(),
                    "method": "circle_pose", "residual": float(resid)}
    return best


def has_plane_pose(calib):
    """True if the calibration carries a recovered plane pose (R, t)."""
    return "R" in calib and "t" in calib


def homography_from_points(image_pts, world_pts):
    """Planar homography from >=4 image<->world (z=0) correspondences."""
    H, _ = cv2.findHomography(np.asarray(image_pts, dtype=np.float64),
                              np.asarray(world_pts, dtype=np.float64)[:, :2])
    return H


def plate_calibration(circle, width, height, plate_radius_m=0.03, focal=None):
    """Build a top-down plate calibration from a detected circle/ellipse.

    Assumes a near-overhead camera: focal defaults to `pose.default_intrinsics`
    and the camera-to-floor distance is inferred from the projected plate radius.
    The tilt of the felt ellipse is not corrected, so XY/Z are approximate.
    """
    f = float(focal if focal is not None else max(width, height))
    r = max(float(circle["radius"]), 1e-6)
    scale = float(plate_radius_m) / r
    calib = {
        "cx": float(circle["cx"]),
        "cy": float(circle["cy"]),
        "radius_px": r,
        "radius_m": float(plate_radius_m),
        "scale_m_per_px": scale,
        "focal": f,
        "camera_distance_m": f * float(plate_radius_m) / r,
        "width": int(width),
        "height": int(height),
        "K": default_intrinsics(int(width), int(height)).tolist(),
        "source": circle.get("source", "hough_circle"),
    }
    if all(key in circle for key in ("major_px", "minor_px", "angle")):
        pose = pose_from_circle(circle, np.asarray(calib["K"], dtype=np.float64),
                                plate_radius_m)
        if pose is not None and pose["residual"] < 1e-2:
            calib["R"] = pose["R"].tolist()
            calib["t"] = pose["t"].tolist()
            calib["normal"] = pose["normal"].tolist()
            calib["pose_residual"] = float(pose["residual"])
            calib["method"] = "circle_pose"
    return calib


def calibrate_plate(bgr, plate_radius_m=0.03, focal=None):
    """Detect the plate (felt ellipse preferred) and return a calibration dict."""
    height, width = bgr.shape[:2]
    circle = detect_plate_ellipse(bgr)
    if circle is None:
        circle = detect_plate_circle(bgr)
    if circle is None or not _plausible_circle(circle, width, height):
        return None
    return plate_calibration(circle, width, height,
                             plate_radius_m=plate_radius_m, focal=focal)


def image_to_machine_xy(points, calib):
    """Map image points (px) to machine-frame XY (m) about the plate centre."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = (pts[:, 0] - calib["cx"]) * calib["scale_m_per_px"]
    y = -(pts[:, 1] - calib["cy"]) * calib["scale_m_per_px"]
    return np.stack([x, y], axis=1)


def project_camera_point(point_cam, calib):
    """Project a camera-frame 3D point to pixels using the calibration intrinsics."""
    f = float(calib["focal"])
    z = max(float(point_cam[2]), 1e-9)
    u = f * float(point_cam[0]) / z + calib["width"] / 2.0
    v = f * float(point_cam[1]) / z + calib["height"] / 2.0
    return np.array([u, v], dtype=np.float64)


def tvec_to_machine(tvec, calib):
    """Convert a PnP camera-frame translation to machine-frame XYZ (m).

    XY comes from the top-down plate mapping; Z is height above the floor plane,
    estimated as (camera distance) - (camera-frame depth).
    """
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    uv = project_camera_point(tvec, calib)
    xy = image_to_machine_xy(uv, calib)[0]
    z = float(calib["camera_distance_m"]) - float(tvec[2])
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def camera_to_machine(point_cam, calib):
    """Map a camera-frame 3D point to machine frame using the recovered plane pose."""
    R = np.asarray(calib["R"], dtype=np.float64)
    t = np.asarray(calib["t"], dtype=np.float64)
    return R.T @ (np.asarray(point_cam, dtype=np.float64).reshape(3) - t)


def machine_from_image(points, calib):
    """Back-project image points onto the machine floor plane (z=0)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not has_plane_pose(calib):
        xy = image_to_machine_xy(pts, calib)
        return np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
    K = np.asarray(calib["K"], dtype=np.float64)
    Kinv = np.linalg.inv(K)
    R = np.asarray(calib["R"], dtype=np.float64)
    t = np.asarray(calib["t"], dtype=np.float64)
    Rt = R.T
    out = np.empty((pts.shape[0], 3))
    for i, (u, v) in enumerate(pts):
        d = Kinv @ np.array([u, v, 1.0])
        denom = float((Rt @ d)[2])
        lam = float((Rt @ t)[2]) / denom if abs(denom) > 1e-12 else 0.0
        out[i] = Rt @ (lam * d - t)
    return out


def save_calibration(calib, path):
    """Persist a calibration dict as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)


def load_calibration(path):
    """Load a calibration dict from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
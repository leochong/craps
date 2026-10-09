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


def plate_calibration(circle, width, height, plate_radius_m=0.03, focal=None):
    """Build a top-down plate calibration from a detected circle/ellipse.

    Assumes a near-overhead camera: focal defaults to `pose.default_intrinsics`
    and the camera-to-floor distance is inferred from the projected plate radius.
    The tilt of the felt ellipse is not corrected, so XY/Z are approximate.
    """
    f = float(focal if focal is not None else max(width, height))
    r = max(float(circle["radius"]), 1e-6)
    scale = float(plate_radius_m) / r
    return {
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


def save_calibration(calib, path):
    """Persist a calibration dict as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)


def load_calibration(path):
    """Load a calibration dict from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
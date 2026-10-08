import cv2
import numpy as np

_PIP_LAYOUT = {
    1: [(0.0, 0.0)],
    2: [(-0.25, -0.25), (0.25, 0.25)],
    3: [(-0.25, -0.25), (0.0, 0.0), (0.25, 0.25)],
    4: [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)],
    5: [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25), (0.0, 0.0)],
    6: [(-0.25, -0.25), (-0.25, 0.0), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.0), (0.25, 0.25)],
}

_FACE_Z = 0.5


def pip_layout_3d(face_value):
    pts = _PIP_LAYOUT[face_value]
    return np.array([[x, y, _FACE_Z] for x, y in pts], dtype=np.float64)


def default_intrinsics(width, height):
    f = float(max(width, height))
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def detect_die_candidates(gray):
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if 0.6 < aspect < 1.6:
            cands.append((x, y, w, h))
    return cands


def extract_pips(gray, bbox, pad=2):
    x, y, w, h = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    roi = gray[y0:y1, x0:x1]

    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pip_area = 0.03 * w * h
    min_area = max(4.0, pip_area * 0.2)
    max_area = pip_area * 1.2
    pips = []
    for c in contours:
        a = cv2.contourArea(c)
        if min_area < a < max_area:
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"] + x0
                cy = m["m01"] / m["m00"] + y0
                pips.append((cx, cy))
    pips.sort(key=lambda p: (-p[1], p[0]))
    return np.array(pips, dtype=np.float64).reshape(-1, 2)


def detect_dice_and_pips(gray):
    results = []
    for bbox in detect_die_candidates(gray):
        pips = extract_pips(gray, bbox)
        if pips.size == 0:
            continue
        results.append({"bbox": bbox, "pips": pips, "face_value": _infer_face_value(pips)})
    return results


def _infer_face_value(pips):
    n = len(pips)
    if n in _PIP_LAYOUT:
        return n
    return None


def solve_die_pose(pips, face_value, K, dist=None):
    if face_value not in _PIP_LAYOUT:
        return None
    model = pip_layout_3d(face_value)
    if model.shape[0] != pips.shape[0] or model.shape[0] < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        model, pips, K, dist if dist is not None else np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    pts_proj, _ = cv2.projectPoints(model, rvec, tvec, K, dist if dist is not None else np.zeros(5))
    resid = np.mean(np.linalg.norm(pts_proj.reshape(-1, 2) - pips, axis=1))
    return {"rvec": rvec, "tvec": tvec, "R": rmat, "residual_px": float(resid)}
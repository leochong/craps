import cv2
import numpy as np

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
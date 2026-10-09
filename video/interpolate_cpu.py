import numpy as np
import cv2


def compute_flow(prev_gray, cur_gray, method="dis"):
    if method == "dis":
        inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return inst.calc(prev_gray, cur_gray, None)
    return cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp(img, flow, scale):
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + scale * flow[..., 0]
    map_y = grid_y + scale * flow[..., 1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def interpolate_frame(prev, cur, t, flow):
    prev_w = warp(prev, flow, -t)
    cur_w = warp(cur, flow, 1.0 - t)
    return cv2.addWeighted(prev_w, 1.0 - t, cur_w, t, 0.0)


class CpuInterpolator:
    def __init__(self, factor, method="dis"):
        self.factor = max(1, int(factor))
        self.method = method

    def stream(self, frames):
        it = iter(frames)
        try:
            prev = next(it)
        except StopIteration:
            return
        yield prev
        for cur in it:
            if self.factor > 1:
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
                flow = compute_flow(prev_gray, cur_gray, self.method)
                for k in range(1, self.factor):
                    t = k / self.factor
                    yield interpolate_frame(prev, cur, t, flow)
            yield cur
            prev = cur
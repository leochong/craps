import cv2


def upscale(frame, size, interpolation=cv2.INTER_LANCZOS4):
    return cv2.resize(frame, size, interpolation=interpolation)


class CpuUpscaler:
    def __init__(self, size):
        self.size = size

    def __call__(self, frame):
        return upscale(frame, self.size)
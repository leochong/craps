import cv2


class FrameSource:
    def __init__(self, path, target_size=None):
        self.path = path
        self.target_size = target_size
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self, idx=None):
        if idx is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self.cap.read()
        if not ok:
            return None
        if self.target_size is not None:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        return frame

    def frames(self, start=0, end=None, step=1):
        end = self.frame_count if end is None else min(end, self.frame_count)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        i = start
        while i < end:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            yield i, frame
            i += 1
            if step > 1 and i < end:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
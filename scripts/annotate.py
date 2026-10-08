import argparse
import os

import cv2

from m1 import pose as pose_mod
from m1.frame_grab import FrameSource
from m1.roll_annotation import DieMark, Roll, RollSet, validate

INITIAL_COLOR = (0, 255, 0)
FINAL_COLOR = (255, 128, 0)
ACTIVE_COLOR = (0, 255, 255)

_FACE_KEYS = {ord(str(n)): n for n in range(1, 7)}


class Annotator:
    def __init__(self, video, out_path, scale=1.0):
        self.scale = scale if scale != 1.0 else None
        self.probe = FrameSource(video)
        self.out_path = out_path

        self.rolls = RollSet(video)
        self.mode = "INITIAL"
        self.frame_idx = 0
        self.frame = None
        self.initial = {"frame": None, "dice": []}
        self.final = {"frame": None, "dice": []}
        self.drag_start = None

        cv2.namedWindow("Roll Annotator")
        cv2.setMouseCallback("Roll Annotator", self._mouse)

    def _active(self):
        return self.initial if self.mode == "INITIAL" else self.final

    def _s(self, v):
        return int(v * self.scale) if self.scale else v

    def _from_display(self, v):
        return int(v / self.scale) if self.scale else v

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x0, y0 = self.drag_start
            self.drag_start = None
            x0, y0 = self._from_display(x0), self._from_display(y0)
            x1, y1 = self._from_display(x), self._from_display(y)
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w >= 3 and h >= 3:
                self._active()["dice"].append(DieMark((x, y, w, h)))

    def _goto(self, idx):
        idx = max(0, min(idx, self.probe.frame_count - 1))
        self.frame_idx = idx
        raw = self.probe.read(idx)
        if raw is None:
            return
        self.frame = cv2.resize(raw, (self._s(raw.shape[1]), self._s(raw.shape[0])), interpolation=cv2.INTER_AREA) if self.scale else raw

    def _auto_detect(self):
        raw = self.probe.read(self.frame_idx)
        if raw is None:
            return
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        detections = pose_mod.detect_dice_and_pips(gray)
        for d in detections:
            self._active()["dice"].append(DieMark(tuple(int(v) for v in d["bbox"]), d.get("face_value")))
        self._active()["frame"] = self.frame_idx

    def _draw_rect(self, out, mark, color, thickness=2):
        x, y, w, h = mark.rect
        p1 = (self._s(x), self._s(y))
        p2 = (self._s(x + w), self._s(y + h))
        cv2.rectangle(out, p1, p2, color, thickness)
        label = f"{mark.face}" if mark.face is not None else "?"
        cv2.putText(out, label, (p1[0], p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def _draw(self):
        out = self.frame.copy()
        for i, m in enumerate(self._active()["dice"]):
            color = ACTIVE_COLOR if i == len(self._active()["dice"]) - 1 else (INITIAL_COLOR if self.mode == "INITIAL" else FINAL_COLOR)
            self._draw_rect(out, m, color)
        if self.mode == "FINAL":
            for m in self.initial["dice"]:
                self._draw_rect(out, m, INITIAL_COLOR, 1)

        banner = f"{self.mode}  frame {self.frame_idx}/{self.probe.frame_count - 1}"
        cv2.putText(out, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        help_line = "LMB-drag:box  1-6:face  a:auto  n/p:+/-1  N/P:+/-30  i/f:mode  d:undo  s:save  esc:quit"
        cv2.putText(out, help_line, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    def _assign_face(self, n):
        dice = self._active()["dice"]
        if dice:
            dice[-1].face = n

    def _undo(self):
        dice = self._active()["dice"]
        if dice:
            dice.pop()

    def _save(self):
        roll = Roll(
            id=self.rolls.next_id(),
            initial_frame=self.initial["frame"],
            final_frame=self.final["frame"],
            initial_dice=self.initial["dice"],
            final_dice=self.final["dice"],
        )
        roll.compute_sum()
        errors = validate(roll)
        if errors:
            print(f"cannot save roll: {errors}")
            return None
        self.rolls.append(roll)
        self.rolls.save(self.out_path)
        print(f"saved {roll.id}  f{roll.initial_frame}->f{roll.final_frame}  sum={roll.sum}")
        self.initial = {"frame": None, "dice": []}
        self.final = {"frame": None, "dice": []}
        return roll

    def _handle_key(self, key):
        if key == ord("a"):
            self._auto_detect()
        elif key in _FACE_KEYS:
            self._assign_face(_FACE_KEYS[key])
        elif key == ord("n"):
            self._goto(self.frame_idx + 1)
        elif key == ord("p"):
            self._goto(self.frame_idx - 1)
        elif key == ord("N"):
            self._goto(self.frame_idx + 30)
        elif key == ord("P"):
            self._goto(self.frame_idx - 30)
        elif key == ord("i"):
            self._active()["frame"] = self.frame_idx
            self.mode = "INITIAL"
            print(f"initial frame = {self.frame_idx}")
        elif key == ord("f"):
            if not self.initial["dice"]:
                print("mark initial dice before switching to FINAL")
            else:
                self.final["frame"] = self.frame_idx
                self.mode = "FINAL"
                print(f"final frame = {self.frame_idx}")
        elif key == ord("d"):
            self._undo()
        elif key == ord("s"):
            self._save()
        elif key == ord("u"):
            self.frame_idx = self.initial["frame"] if self.initial["frame"] is not None else self.frame_idx
            self._goto(self.frame_idx)

    def run(self):
        self._goto(0)
        while True:
            cv2.imshow("Roll Annotator", self._draw())
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            self._handle_key(key)

        cv2.destroyAllWindows()
        self.rolls.save(self.out_path)
        print(f"total rolls annotated: {len(self.rolls.rolls)}")


def main():
    ap = argparse.ArgumentParser(description="HITL roll annotator: initial then final die position")
    ap.add_argument("--video", default="IMG_2111.MOV")
    ap.add_argument("--out", default="annotations/rolls.json")
    ap.add_argument("--scale", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Annotator(args.video, args.out, scale=args.scale).run()


if __name__ == "__main__":
    main()
import json
import os
from dataclasses import dataclass, field


@dataclass
class DieMark:
    rect: tuple
    face: int = None

    def to_dict(self):
        return {"rect": [int(v) for v in self.rect], "face": self.face}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["rect"]), d.get("face"))


@dataclass
class Roll:
    id: str
    initial_frame: int
    final_frame: int
    initial_dice: list = field(default_factory=list)
    final_dice: list = field(default_factory=list)
    sum: int = None

    def to_dict(self):
        return {
            "id": self.id,
            "initial_frame": self.initial_frame,
            "final_frame": self.final_frame,
            "initial_dice": [m.to_dict() for m in self.initial_dice],
            "final_dice": [m.to_dict() for m in self.final_dice],
            "sum": self.sum,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d["id"],
            initial_frame=d["initial_frame"],
            final_frame=d["final_frame"],
            initial_dice=[DieMark.from_dict(m) for m in d.get("initial_dice", [])],
            final_dice=[DieMark.from_dict(m) for m in d.get("final_dice", [])],
            sum=d.get("sum"),
        )

    def compute_sum(self):
        faces = [m.face for m in self.final_dice if m.face is not None]
        self.sum = sum(faces) if len(faces) == len(self.final_dice) and faces else None
        return self.sum


def validate(roll):
    errors = []
    if not roll.initial_dice:
        errors.append("no initial dice marked")
    if not roll.final_dice:
        errors.append("no final dice marked")
    if any(m.face is None for m in roll.initial_dice + roll.final_dice):
        errors.append("some dice missing a face value")
    if roll.final_frame <= roll.initial_frame:
        errors.append("final_frame must be after initial_frame")
    return errors


class RollSet:
    def __init__(self, video, rolls=None):
        self.video = video
        self.rolls = rolls or []

    def append(self, roll):
        self.rolls.append(roll)

    def next_id(self):
        return f"roll_{len(self.rolls) + 1:04d}"

    def to_dict(self):
        return {"video": self.video, "rolls": [r.to_dict() for r in self.rolls]}

    @classmethod
    def from_dict(cls, d):
        return cls(video=d.get("video"), rolls=[Roll.from_dict(r) for r in d.get("rolls", [])])

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
"""Sequence index, frame loading, split assignment and stream specification.

A *sequence* is an ordered list of frame image paths recorded by one front camera
in one drive, plus metadata attributes (weather, timeofday, scene, ...).
Sequences are the independent experimental units.

A *stream* is what the online monitor observes: a pre-change segment followed
(optionally) by a post-change segment.  Change point ``tau`` is the 0-based index
of the first post-change frame in the stream.  Clean streams have tau=None.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np


@dataclass
class Sequence:
    seq_id: str
    dataset: str
    frames: list[str]
    attrs: dict = field(default_factory=dict)
    fps: float = 5.0


def load_index(path: str) -> list[Sequence]:
    with open(path) as f:
        raw = json.load(f)
    return [Sequence(**r) for r in raw]


def save_index(seqs: list[Sequence], path: str):
    with open(path, "w") as f:
        json.dump([asdict(s) for s in seqs], f, indent=1)


def stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def load_frame(path: str, size_hw: tuple[int, int]) -> np.ndarray:
    """Read an RGB frame and resize to (H, W). Returns uint8 HxWx3."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def resize(im: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(im, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)


@dataclass
class Segment:
    seq_id: str
    start: int
    end: int  # exclusive
    corruption: str | None = None
    severity: int | None = None

    @property
    def clip_id(self) -> str:
        return f"{self.seq_id}__clean" if self.corruption is None else f"{self.seq_id}__{self.corruption}_s{self.severity}"


@dataclass
class StreamSpec:
    stream_id: str
    split: str           # 'calib' | 'dev' | 'test'
    kind: str            # 'clean' | 'synthetic' | 'natural' | 'null_splice'
    shift: str           # e.g. 'fog', 'night', 'none'
    severity: int | None
    segments: list[Segment]
    tau: int | None
    unit: str            # independent experimental unit id (sequence id of the pre-change part)

    def to_json(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_json(d):
        d = dict(d)
        d["segments"] = [Segment(**s) for s in d["segments"]]
        return StreamSpec(**d)


def save_streams(streams: list[StreamSpec], path: str):
    with open(path, "w") as f:
        json.dump([s.to_json() for s in streams], f, indent=1)


def load_streams(path: str) -> list[StreamSpec]:
    with open(path) as f:
        return [StreamSpec.from_json(d) for d in json.load(f)]

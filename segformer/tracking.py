"""Temporal smoothing: turn per-frame detections into stable tracks.

SegFormer-B0 output flickers frame to frame (borderline confidence, borderline
geometry). A detection must survive ``min_hits`` frames to be reported, and
lingers ``max_misses`` frames after it's lost — so a door pointed at steadily
reads as one continuous detection, not a strobe.

Matching is per-class by mask IoU. For this task each class is usually a single
object, so this is mostly hysteresis with light box/mask smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from segformer.segmenter import Instance, SegResult


def _boxes_disjoint(p: tuple, q: tuple) -> bool:
    return p[2] < q[0] or q[2] < p[0] or p[3] < q[1] or q[3] < p[1]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    return inter / np.count_nonzero(a | b)


@dataclass
class _Track:
    track_id: int
    label: int
    mask: np.ndarray
    box: tuple[float, float, float, float]
    score: float
    hits: int = 1
    misses: int = 0

    def as_instance(self) -> Instance:
        return Instance(self.track_id, self.label, self.score, self.mask, self.box)


@dataclass
class Tracker:
    iou_match: float = 0.3
    min_hits: int = 3
    max_misses: int = 6
    score_ema: float = 0.5
    _tracks: list[_Track] = field(default_factory=list)
    _next_id: int = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, result: SegResult) -> SegResult:
        dets = list(result.instances)
        used = set()

        for tr in self._tracks:
            best, best_iou = -1, self.iou_match
            for i, d in enumerate(dets):
                if i in used or d.label != tr.label:
                    continue
                if _boxes_disjoint(tr.box, d.box_xyxy):
                    continue
                v = _iou(tr.mask, d.mask)
                if v >= best_iou:
                    best, best_iou = i, v
            if best >= 0:
                d = dets[best]
                used.add(best)
                tr.mask, tr.box = d.mask, d.box_xyxy
                tr.score = self.score_ema * d.score + (1 - self.score_ema) * tr.score
                tr.hits += 1
                tr.misses = 0
            else:
                tr.misses += 1

        for i, d in enumerate(dets):
            if i not in used:
                self._tracks.append(_Track(self._next_id, d.label, d.mask, d.box_xyxy, d.score))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        out = SegResult(frame_id=result.frame_id, timestamp_ns=result.timestamp_ns)
        out.instances = [
            t.as_instance() for t in self._tracks
            if t.hits >= self.min_hits and t.misses <= self.max_misses // 2
        ]
        return out

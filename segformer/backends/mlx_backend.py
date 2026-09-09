"""SegFormer semantic segmentation on MLX (Apple GPU). Primary backend.

For each target ADE class, threshold its probability map (argmax over-includes
neighbouring wall for 'door'), take connected components, then:
  * depth gate for door-like classes — must be a door-sized, planar, roughly
    vertical structure (tolerant of a door only partly in frame — the USB
    tether is short)
No temporal state here — see segformer.tracking for flicker suppression.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from segformer.frame import Frame
from segformer.model.config import ADE20K_TARGETS
from segformer.segmenter import Instance, SegResult, Segmenter

_DOORLIKE = {14, 58}                       # door, screen door -> depth gate
_PROB_THRESH = {14: 0.85, 58: 0.85}       # high: isolate the leaf from the wall
_DEFAULT_PROB_THRESH = 0.45               # railing, bannister
_MIN_COMP = 150                           # ignore components smaller than this (px)


class MlxSegmenter(Segmenter):
    name = "mlx"

    def __init__(self, model: str | None = None, *, min_area: int = 600,
                 use_depth: bool = True, smooth: bool = True,
                 targets: dict[int, str] | None = None):
        self.model_name = model
        self.min_area = min_area
        self.use_depth = use_depth
        self.targets = targets or ADE20K_TARGETS
        self._model = None
        self._tracker = None
        if smooth:
            from segformer.tracking import Tracker

            self._tracker = Tracker()

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()

    def load(self, checkpoint: str | None = None) -> None:
        if self._model is not None:
            return
        from segformer.model import load_model

        self._model = load_model(self.model_name, checkpoint=checkpoint)

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        import cv2

        target_ids = list(self.targets)
        prob = self._model.class_probs(frame.rgb, target_ids)     # (H,W,T)
        res = SegResult(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns)
        next_id = 1
        k3 = np.ones((3, 3), np.uint8)
        _points = None  # backproject lazily, once, only if a door candidate exists

        for ti, ade_id in enumerate(target_ids):
            thresh = _PROB_THRESH.get(ade_id, _DEFAULT_PROB_THRESH)
            binm = (prob[:, :, ti] > thresh).astype(np.uint8)
            if not binm.any():
                continue
            binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k3)
            nlab, lab, stats, _cent = cv2.connectedComponentsWithStats(binm)
            cand = [c for c in range(1, nlab) if stats[c, cv2.CC_STAT_AREA] >= _MIN_COMP]
            if not cand:
                continue
            pmap = prob[:, :, ti]

            if self.use_depth and ade_id in _DOORLIKE:
                from segformer.geometry import consolidate_door, door_gate

                if _points is None:
                    from segformer.geometry import backproject

                    _points = backproject(frame.depth, frame.intrinsics)
                points, valid = _points

                merged = consolidate_door(lab, cand, stats, frame.depth)
                if int(merged.sum()) < self.min_area:
                    continue
                if not door_gate(merged, points, valid, frame.intrinsics).keep:
                    continue
                res.instances.append(self._instance(next_id, ade_id, merged, pmap))
                next_id += 1
            else:
                for c in cand:
                    if stats[c, cv2.CC_STAT_AREA] < self.min_area:
                        continue
                    res.instances.append(self._instance(next_id, ade_id, lab == c, pmap))
                    next_id += 1

        if self._tracker is not None:
            res = self._tracker.update(res)
        return res, state

    @staticmethod
    def _instance(iid: int, label: int, mask: np.ndarray, pmap: np.ndarray) -> Instance:
        ys, xs = np.where(mask)
        return Instance(
            instance_id=iid,
            label=label,
            score=float(pmap[mask].mean()),
            mask=mask,
            box_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        )

    def close(self) -> None:
        self._model = None

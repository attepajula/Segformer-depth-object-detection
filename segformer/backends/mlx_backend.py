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

        points = valid = None
        if self.use_depth:
            from segformer.geometry import backproject

            points, valid = backproject(frame.depth, frame.intrinsics)

        res = SegResult(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns)
        next_id = 1
        for ti, ade_id in enumerate(target_ids):
            thresh = _PROB_THRESH.get(ade_id, _DEFAULT_PROB_THRESH)
            cls_mask = (prob[:, :, ti] > thresh).astype(np.uint8)
            if not cls_mask.any():
                continue
            cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, comp = cv2.connectedComponents(cls_mask)
            for c in range(1, n):
                m = comp == c
                if int(m.sum()) < self.min_area:
                    continue
                if self.use_depth and ade_id in _DOORLIKE:
                    from segformer.geometry import door_gate

                    g = door_gate(m, points, valid, frame.intrinsics)
                    if not g.keep:
                        continue
                ys, xs = np.where(m)
                res.instances.append(
                    Instance(
                        instance_id=next_id,
                        label=ade_id,
                        score=float(prob[m, ti].mean()),
                        mask=m,
                        box_xyxy=(float(xs.min()), float(ys.min()),
                                  float(xs.max()), float(ys.max())),
                    )
                )
                next_id += 1

        if self._tracker is not None:
            res = self._tracker.update(res)
        return res, state

    def close(self) -> None:
        self._model = None

"""SegFormer semantic segmentation on MLX (Apple GPU). Primary backend.

Pretrained on ADE20K; surface the door / railing / bannister / screen-door
classes as instances (one per connected component). Two false-positive filters:
  * ``score_thresh`` — drop low-confidence components
  * depth gate (``use_depth``) — a "door" must be a door-sized planar vertical
    surface; wall-sized / non-planar / floor-ish blobs are rejected.
No temporal state — SegFormer is per-frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from segformer.frame import Frame
from segformer.model.config import ADE20K_TARGETS
from segformer.segmenter import Instance, SegResult, Segmenter

# ADE ids that get the door depth-gate
_DOORLIKE = {14, 58}


class MlxSegmenter(Segmenter):
    name = "mlx"

    def __init__(self, model: str | None = None, *, min_area: int = 600,
                 score_thresh: float = 0.45, use_depth: bool = True,
                 targets: dict[int, str] | None = None):
        self.model_name = model
        self.min_area = min_area
        self.score_thresh = score_thresh
        self.use_depth = use_depth
        self.targets = targets or ADE20K_TARGETS
        self._model = None

    def load(self, checkpoint: str | None = None) -> None:
        if self._model is not None:
            return
        from segformer.model import load_model

        self._model = load_model(self.model_name, checkpoint=checkpoint)

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        import cv2

        target_ids = list(self.targets)
        lab, prob = self._model.predict(frame.rgb, target_ids)   # (H,W), (H,W,T)

        points = valid = None
        if self.use_depth:
            from segformer.geometry import backproject

            points, valid = backproject(frame.depth, frame.intrinsics)

        res = SegResult(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns)
        next_id = 1
        for ti, ade_id in enumerate(target_ids):
            cls_mask = (lab == ade_id).astype(np.uint8)
            if not cls_mask.any():
                continue
            n, comp = cv2.connectedComponents(cls_mask)
            for c in range(1, n):
                m = comp == c
                if int(m.sum()) < self.min_area:
                    continue
                score = float(prob[m, ti].mean())
                if score < self.score_thresh:
                    continue
                if self.use_depth and ade_id in _DOORLIKE:
                    from segformer.geometry import door_gate

                    if not door_gate(m, points, valid, frame.intrinsics).keep:
                        continue
                ys, xs = np.where(m)
                res.instances.append(
                    Instance(
                        instance_id=next_id,
                        label=ade_id,
                        score=score,
                        mask=m,
                        box_xyxy=(float(xs.min()), float(ys.min()),
                                  float(xs.max()), float(ys.max())),
                    )
                )
                next_id += 1
        return res, state

    def close(self) -> None:
        self._model = None

"""SegFormer semantic segmentation on MLX (Apple GPU). Primary backend.

Pretrained on ADE20K; we surface the door / railing / bannister / screen-door
classes as instances (one per connected component) and drop the rest.
No temporal state — SegFormer is per-frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from segformer.frame import Frame
from segformer.model.config import ADE20K_TARGETS
from segformer.segmenter import Instance, SegResult, Segmenter


class MlxSegmenter(Segmenter):
    name = "mlx"

    def __init__(self, model: str | None = None, min_area: int = 400,
                 targets: dict[int, str] | None = None):
        self.model_name = model            # registry name; None => models.toml default
        self.min_area = min_area
        self.targets = targets or ADE20K_TARGETS
        self._model = None

    def load(self, checkpoint: str | None = None) -> None:
        if self._model is not None:
            return
        from segformer.model import load_model

        self._model = load_model(self.model_name, checkpoint=checkpoint)

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        import cv2
        import mlx.core as mx

        logits = self._model.logits(frame.rgb)          # (H, W, 150) mlx
        prob = np.array(mx.softmax(logits, axis=-1))
        lab = prob.argmax(-1).astype(np.int32)

        res = SegResult(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns)
        next_id = 1
        for ade_id, _name in self.targets.items():
            cls_mask = lab == ade_id
            if not cls_mask.any():
                continue
            n, comp = cv2.connectedComponents(cls_mask.astype(np.uint8))
            for c in range(1, n):
                m = comp == c
                area = int(m.sum())
                if area < self.min_area:
                    continue
                ys, xs = np.where(m)
                res.instances.append(
                    Instance(
                        instance_id=next_id,
                        label=ade_id,
                        score=float(prob[m, ade_id].mean()),
                        mask=m,
                        box_xyxy=(float(xs.min()), float(ys.min()),
                                  float(xs.max()), float(ys.max())),
                    )
                )
                next_id += 1
        return res, state

    def close(self) -> None:
        self._model = None

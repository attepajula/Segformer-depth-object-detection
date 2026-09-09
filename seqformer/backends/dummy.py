"""No-model segmenter: emits one instance covering the valid-depth region.

Not a real model. Exists so the pipeline (consumer -> segmenter -> overlay) can
be wired and tested before SeqFormer lands.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from seqformer.frame import Frame
from seqformer.segmenter import Instance, SegResult, Segmenter


class DummySegmenter(Segmenter):
    name = "dummy"

    def load(self, checkpoint: str | None = None) -> None:
        pass

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        mask = frame.depth_valid_mask
        res = SegResult(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns)
        if mask.any():
            ys, xs = np.where(mask)
            res.instances.append(
                Instance(
                    instance_id=1,
                    label=0,
                    score=1.0,
                    mask=mask,
                    box_xyxy=(float(xs.min()), float(ys.min()),
                              float(xs.max()), float(ys.max())),
                )
            )
        return res, state

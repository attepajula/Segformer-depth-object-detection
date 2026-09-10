"""SegFormer semantic segmentation on MLX (Apple GPU). Primary backend.

Produces per-class probability maps; the shared pipeline in
``segformer.backends.postprocess`` turns them into a SegResult (threshold,
connected components, depth consolidation + gate, temporal tracker).
"""

from __future__ import annotations

from typing import Any

from segformer.backends.postprocess import probs_to_result
from stream.frame import Frame
from segformer.labels import ADE20K_TARGETS
from segformer.segmenter import SegResult, Segmenter


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
        try:
            from segformer.model import load_model
        except ImportError as e:
            raise ImportError(
                "mlx backend needs mlx:  uv pip install -e '.[mlx]'\n"
                "or use a different backend:  --backend torch  /  SEGFORMER_BACKEND=torch"
            ) from e
        self._model = load_model(self.model_name, checkpoint=checkpoint)

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        ids = list(self.targets)
        prob = self._model.class_probs(frame.rgb, ids)          # (H,W,T) numpy
        res = probs_to_result(frame, prob, ids, min_area=self.min_area,
                              use_depth=self.use_depth, tracker=self._tracker)
        return res, state

    def close(self) -> None:
        self._model = None

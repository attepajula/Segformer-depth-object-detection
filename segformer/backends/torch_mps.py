"""SegFormer on PyTorch / Apple MPS. STUB — secondary backend.

The MLX backend (``mlx_backend.py``) is primary and validated bit-exact against
HF transformers. This one exists as a fallback / cross-check path.

Plan:
  * ``from transformers import SegformerForSemanticSegmentation`` on
    ``nvidia/segformer-b0-finetuned-ade-512-512``.
  * ``model.to("mps")``; ``SegformerImageProcessor`` for preprocessing.
  * ``infer``: logits -> upsample -> argmax -> keep ADE ids in
    ``ADE20K_TARGETS`` -> connected components -> Instance list
    (identical post-process to ``MlxSegmenter``; factor it out when built).
"""

from __future__ import annotations

from typing import Any

from segformer.frame import Frame
from segformer.segmenter import SegResult, Segmenter


class TorchMpsSegmenter(Segmenter):
    name = "torch-mps"

    def __init__(self, size: str = "b0"):
        self.size = size
        self._model = None

    def load(self, checkpoint: str | None = None) -> None:
        raise NotImplementedError("TorchMpsSegmenter — use --backend mlx (primary)")

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        raise NotImplementedError("TorchMpsSegmenter")

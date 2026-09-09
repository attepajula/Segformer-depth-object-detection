"""SeqFormer on MLX (Apple GPU, unified memory). STUB.

Plan / open problems:
  * ``mx.array`` from the frame's numpy buffers is zero-copy on unified memory.
  * Backbone: ResNet-50 in ``mlx.nn`` (straightforward) or Swin (more work).
  * MS-deform-attn: no MLX op. Two routes —
      1. pure-MLX: gather + bilinear sample at offset locations + weighted sum
         (works, not fused, memory-heavy);
      2. ``mx.fast.metal_kernel`` custom Metal kernel (fast, more effort).
  * torchvision ops (NMS, box IoU, roi ops) have no MLX equivalent — reimplement
    (batched NMS in pure MLX is short).
  * Weights: convert the PyTorch SeqFormer state dict -> MLX (key remap,
    transpose conv/linear weights as needed).
  * ``mx.eval`` at the right points to bound the lazy graph per frame.

Nothing below is implemented.
"""

from __future__ import annotations

from typing import Any

from seqformer.frame import Frame
from seqformer.segmenter import SegResult, Segmenter


class MlxSegmenter(Segmenter):
    name = "mlx"

    def __init__(self, clip_len: int = 5, score_thresh: float = 0.3):
        self.clip_len = clip_len
        self.score_thresh = score_thresh
        self._model = None

    def load(self, checkpoint: str | None = None) -> None:
        raise NotImplementedError("MlxSegmenter.load — see module docstring")

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        raise NotImplementedError("MlxSegmenter.infer")

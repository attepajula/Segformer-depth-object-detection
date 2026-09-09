"""SeqFormer on PyTorch, Apple GPU via the MPS backend. STUB.

Plan:
  * device = ``torch.device("mps")`` (fallback "cpu"); set
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` for ops MPS lacks.
  * Backbone: ResNet-50 or Swin-T from torchvision, C3-C5 feature pyramid.
  * Deformable-DETR encoder/decoder. MS-deform-attn: the fused CUDA op does not
    exist on MPS — use the pure-PyTorch fallback (bilinear ``grid_sample`` at
    the sampling locations + weighted sum). Slower, correct.
  * SeqFormer instance head: per-clip query, mask head via dot-product between
    per-instance embeddings and the mask feature map.
  * Checkpoint: load the official SeqFormer PyTorch weights directly.
  * Rolling state: a deque of the last N frames' backbone features for the clip
    attention.

Nothing below is implemented.
"""

from __future__ import annotations

from typing import Any

from seqformer.frame import Frame
from seqformer.segmenter import SegResult, Segmenter


class TorchMpsSegmenter(Segmenter):
    name = "torch-mps"

    def __init__(self, clip_len: int = 5, score_thresh: float = 0.3):
        self.clip_len = clip_len
        self.score_thresh = score_thresh
        self._model = None
        self._device = None

    def load(self, checkpoint: str | None = None) -> None:
        raise NotImplementedError("TorchMpsSegmenter.load — see module docstring")

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        raise NotImplementedError("TorchMpsSegmenter.infer")

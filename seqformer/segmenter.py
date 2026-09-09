"""Framework-agnostic segmentation interface.

Everything downstream (CLI, overlay, tracking, eval) talks to :class:`Segmenter`
and :class:`SegResult` only. A PyTorch/MPS implementation and an MLX one are
drop-in interchangeable — pick at construction time via
``seqformer.backends.make_segmenter``.

SeqFormer is a *video* instance segmentation model: it attends across a short
clip of frames. ``infer`` therefore takes the current frame plus the model's
own rolling state, and returns updated state alongside the result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seqformer.frame import Frame


@dataclass
class Instance:
    """One segmented instance in one frame."""

    instance_id: int          # stable across the clip (tracking id)
    label: int                # class index
    score: float              # confidence in [0, 1]
    mask: np.ndarray          # (H, W) bool
    box_xyxy: tuple[float, float, float, float]


@dataclass
class SegResult:
    frame_id: int
    timestamp_ns: int
    instances: list[Instance] = field(default_factory=list)

    def label_map(self, hw: tuple[int, int]) -> np.ndarray:
        """(H, W) int32; 0 = background, otherwise instance_id. Higher score wins
        on overlap."""
        out = np.zeros(hw, dtype=np.int32)
        for inst in sorted(self.instances, key=lambda i: i.score):
            out[inst.mask] = inst.instance_id
        return out


class Segmenter(ABC):
    """Stateful video-instance segmenter.

    Lifecycle: ``load()`` once, then ``infer()`` per frame, ``reset()`` on a
    stream discontinuity (e.g. replay loop, camera reopen).
    """

    #: short backend name, e.g. "torch-mps", "mlx"
    name: str = "abstract"

    @abstractmethod
    def load(self, checkpoint: str | None = None) -> None:
        """Build the model and move weights to the device. Idempotent."""

    @abstractmethod
    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        """Segment ``frame`` given rolling clip ``state``; return (result, new_state)."""

    def reset(self) -> None:
        """Drop rolling state at a stream boundary. Override if stateful."""

    def close(self) -> None:
        """Release device memory."""

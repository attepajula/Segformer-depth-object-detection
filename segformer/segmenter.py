"""Framework-agnostic segmentation interface.

Everything downstream (CLI, overlay, eval) talks to :class:`Segmenter` and
:class:`SegResult` only. Backends (mlx, torch, dummy) are drop-in
interchangeable — pick at construction time via
``segformer.backends.make_segmenter``.

``infer`` takes the current frame plus an opaque ``state`` and returns an
updated state, so a stateful backend (temporal smoothing, tracking) fits
without changing callers. SegFormer itself is per-frame — its backend ignores
``state``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stream.frame import Frame


@dataclass
class Instance:
    """One segmented region in one frame (a connected component of a class)."""

    instance_id: int          # unique within the frame
    label: int                # ADE20K class id (see model.config.ADE20K_TARGETS)
    score: float              # mean class probability over the mask, [0, 1]
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
    """A frame segmenter.

    Lifecycle: ``load()`` once, then ``infer()`` per frame, ``reset()`` on a
    stream discontinuity (replay loop, camera reopen).
    """

    #: short backend name, e.g. "mlx", "torch", "dummy"
    name: str = "abstract"

    @abstractmethod
    def load(self, checkpoint: str | None = None) -> None:
        """Build the model and move weights to the device. Idempotent."""

    @abstractmethod
    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        """Segment ``frame`` given opaque ``state``; return (result, new_state)."""

    def reset(self) -> None:
        """Drop rolling state at a stream boundary. Override if stateful."""

    def close(self) -> None:
        """Release device memory."""

"""Decoded frame — the consumer-side view of the capture pipeline's contract.

Mirrors ``capture/frame.py`` schema v1. Kept as an independent copy on purpose:
segformer must not import the capture package. If the wire schema bumps, update
``WIRE_SCHEMA_VERSION`` and this file together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(
            fx=float(d["fx"]), fy=float(d["fy"]),
            cx=float(d["cx"]), cy=float(d["cy"]),
            width=int(d["width"]), height=int(d["height"]),
        )


@dataclass
class Frame:
    """One registered RGB-D frame off the wire.

    rgb  : (H, W, 3) uint8, R,G,B
    depth: (H, W) uint16, millimeters, 0 = invalid
    """

    schema_version: int
    frame_id: int
    timestamp_ns: int
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: Intrinsics
    depth_scale: float
    registered: bool
    device: str

    @property
    def hw(self) -> tuple[int, int]:
        return self.depth.shape  # (H, W)

    @property
    def depth_valid_mask(self) -> np.ndarray:
        """Bool (H, W); False where there is no depth measurement."""
        return self.depth > 0

"""The RGB-D frame contract.

Mirrors the capture service's schema v1 (`../xtion`, `capture/frame.py`) — a
deliberately separate copy of a frozen format. If the schema ever bumps, this
file, ``wire.py`` and ``WIRE_SCHEMA_VERSION`` change together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIRE_SCHEMA_VERSION = 1  # bump only on an incompatible frame/recording change


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

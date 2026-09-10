"""Read a recorded RGB-D session folder.

    session/
      meta.json          schema_version, device, depth_scale, intrinsics, ...
      chunk_00000.npz     rgb (N,H,W,3) u8, depth (N,H,W) u16,
      chunk_00001.npz     frame_id (N,) i64, timestamp_ns (N,) i64
      ...
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stream.frame import WIRE_SCHEMA_VERSION

_META = "meta.json"
_CHUNKS = "chunk_*.npz"


class RecordingReader:
    def __init__(self, path: str | Path):
        self.dir = Path(path)
        meta_path = self.dir / _META
        if not meta_path.is_file():
            raise FileNotFoundError(f"no {_META} in {self.dir}")
        self.meta = json.loads(meta_path.read_text())
        ver = self.meta.get("schema_version")
        if ver != WIRE_SCHEMA_VERSION:
            raise ValueError(f"recording schema_version {ver} != {WIRE_SCHEMA_VERSION}")
        self.intrinsics = dict(self.meta["intrinsics"])
        self.device = self.meta.get("device", "replay")
        self.depth_scale = float(self.meta.get("depth_scale", 0.001))
        self._chunks = sorted(self.dir.glob(_CHUNKS))
        if not self._chunks:
            raise FileNotFoundError(f"no {_CHUNKS} in {self.dir}")

    @property
    def frame_count(self) -> int:
        return int(self.meta.get("frame_count", 0))

    def _header(self, frame_id: int, timestamp_ns: int) -> dict:
        return {
            "schema_version": WIRE_SCHEMA_VERSION,
            "frame_id": int(frame_id),
            "timestamp_ns": int(timestamp_ns),
            "depth_scale": self.depth_scale,
            "registered": True,
            "device": f"replay:{self.device}",
            "intrinsics": self.intrinsics,
        }

    def frames(self):
        """Yield (header, rgb, depth) in recorded order."""
        for chunk in self._chunks:
            with np.load(chunk) as z:
                rgb, depth = z["rgb"], z["depth"]
                fid, ts = z["frame_id"], z["timestamp_ns"]
                for i in range(len(fid)):
                    yield self._header(fid[i], ts[i]), rgb[i], depth[i]

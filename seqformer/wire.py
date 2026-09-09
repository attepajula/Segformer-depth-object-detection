"""Decode schema-v1 frames off the ZeroMQ wire.

Wire format (see the capture pipeline's docs/wire-format.md): one single-part
msgpack map per frame, pixel arrays as ``{dtype, shape, data}`` sub-maps with
raw C-contiguous buffers, optionally lz4-compressed.

Decode only — this process never publishes.
"""

from __future__ import annotations

import msgpack
import numpy as np

from seqformer import WIRE_SCHEMA_VERSION
from seqformer.frame import Frame, Intrinsics

try:
    import lz4.block as _lz4
except Exception:  # optional
    _lz4 = None


class WireError(RuntimeError):
    pass


def _unpack_array(d: dict, compression: str) -> np.ndarray:
    raw = d["data"]
    if compression == "lz4":
        if _lz4 is None:
            raise WireError("frame is lz4-compressed but python-lz4 is not installed")
        raw = _lz4.decompress(raw)
    elif compression != "none":
        raise WireError(f"unknown compression {compression!r}")
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(tuple(d["shape"]))


def decode_frame(buf: bytes) -> Frame:
    msg = msgpack.unpackb(buf, raw=False)
    ver = msg.get("schema_version")
    if ver != WIRE_SCHEMA_VERSION:
        raise WireError(f"frame schema_version {ver} != {WIRE_SCHEMA_VERSION}")

    comp = msg.get("compression", "none")
    rgb = _unpack_array(msg["rgb"], comp)
    depth = _unpack_array(msg["depth"], comp)

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise WireError(f"bad rgb {rgb.shape} {rgb.dtype}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise WireError(f"bad depth {depth.shape} {depth.dtype}")
    if not msg.get("registered", False):
        raise WireError("frame is not registered")

    return Frame(
        schema_version=ver,
        frame_id=int(msg["frame_id"]),
        timestamp_ns=int(msg["timestamp_ns"]),
        rgb=rgb,
        depth=depth,
        intrinsics=Intrinsics.from_dict(msg["intrinsics"]),
        depth_scale=float(msg["depth_scale"]),
        registered=True,
        device=str(msg["device"]),
    )


def decode_header(buf: bytes) -> dict:
    """Contract fields only; skips pixel arrays."""
    msg = msgpack.unpackb(buf, raw=False)
    msg.pop("rgb", None)
    msg.pop("depth", None)
    return msg

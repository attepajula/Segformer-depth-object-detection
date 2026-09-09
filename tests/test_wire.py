"""Wire decode must match the capture pipeline's schema v1. The encoder here is
a local mirror of capture/codec.py — if this drifts, the contract broke."""

import msgpack
import numpy as np
import pytest

from seqformer.backends import make_segmenter
from seqformer.wire import WireError, decode_frame, decode_header


def _encode(rgb, depth, *, frame_id=1, device="fake", registered=True, schema_version=1):
    def arr(a):
        a = np.ascontiguousarray(a)
        return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.tobytes()}

    msg = {
        "schema_version": schema_version,
        "frame_id": frame_id,
        "timestamp_ns": 123,
        "depth_scale": 0.001,
        "registered": registered,
        "device": device,
        "intrinsics": {"fx": 570.0, "fy": 570.0, "cx": 320.0, "cy": 240.0,
                       "width": rgb.shape[1], "height": rgb.shape[0]},
        "compression": "none",
        "rgb": arr(rgb),
        "depth": arr(depth),
    }
    return msgpack.packb(msg, use_bin_type=True)


def _sample():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[..., 0] = 11
    rgb[..., 1] = 22
    rgb[..., 2] = 33
    depth = np.full((4, 6), 1500, np.uint16)
    depth[0] = 0
    return rgb, depth


def test_round_trip_and_channel_order():
    rgb, depth = _sample()
    f = decode_frame(_encode(rgb, depth, frame_id=7))
    assert f.frame_id == 7
    assert f.rgb.dtype == np.uint8 and f.depth.dtype == np.uint16
    assert tuple(f.rgb[1, 1]) == (11, 22, 33)
    np.testing.assert_array_equal(f.depth, depth)
    assert f.intrinsics.width == 6 and f.intrinsics.height == 4
    assert f.depth_valid_mask.sum() == 18  # 3 valid rows * 6


def test_header_only_decode():
    rgb, depth = _sample()
    hdr = decode_header(_encode(rgb, depth, frame_id=99))
    assert hdr["frame_id"] == 99 and "rgb" not in hdr


def test_rejects_wrong_schema():
    rgb, depth = _sample()
    with pytest.raises(WireError):
        decode_frame(_encode(rgb, depth, schema_version=2))


def test_rejects_unregistered():
    rgb, depth = _sample()
    with pytest.raises(WireError):
        decode_frame(_encode(rgb, depth, registered=False))


def test_dummy_segmenter_emits_valid_region():
    rgb, depth = _sample()
    frame = decode_frame(_encode(rgb, depth))
    seg = make_segmenter("dummy")
    seg.load()
    res, _ = seg.infer(frame, None)
    assert len(res.instances) == 1
    assert res.instances[0].mask.sum() == 18

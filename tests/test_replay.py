"""Replay: read a recording folder, publish it, decode it back unchanged."""

import json
import threading
import time

import numpy as np
import pytest
import zmq

from stream.recording import RecordingReader
from stream.replay import publish
from stream.wire import decode_frame, encode_frame


def _make_recording(d, n=5):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (n, 8, 12, 3), np.uint8)
    depth = rng.integers(0, 4000, (n, 8, 12), np.uint16)
    np.savez(d / "chunk_00000.npz", rgb=rgb, depth=depth,
             frame_id=np.arange(n, dtype=np.int64),
             timestamp_ns=(np.arange(n) * 33_000_000).astype(np.int64))
    (d / "meta.json").write_text(json.dumps({
        "schema_version": 1, "device": "xtion", "depth_scale": 0.001,
        "registered": True, "frame_count": n,
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 6.0, "cy": 4.0,
                       "width": 12, "height": 8},
    }))
    return rgb, depth


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, (8, 12, 3), np.uint8)
    depth = rng.integers(0, 4000, (8, 12), np.uint16)
    hdr = {"schema_version": 1, "frame_id": 7, "timestamp_ns": 123, "depth_scale": 0.001,
           "registered": True, "device": "replay:xtion",
           "intrinsics": {"fx": 1, "fy": 1, "cx": 6, "cy": 4, "width": 12, "height": 8}}
    for comp in ("none", "lz4"):
        f = decode_frame(encode_frame(hdr, rgb, depth, comp))
        assert f.frame_id == 7 and f.device == "replay:xtion"
        np.testing.assert_array_equal(f.rgb, rgb)
        np.testing.assert_array_equal(f.depth, depth)


def test_reader(tmp_path):
    rgb, depth = _make_recording(tmp_path, n=5)
    r = RecordingReader(tmp_path)
    assert r.frame_count == 5 and r.device == "xtion"
    got = list(r.frames())
    assert len(got) == 5
    h0, rgb0, d0 = got[0]
    assert h0["device"] == "replay:xtion" and h0["registered"] is True
    np.testing.assert_array_equal(rgb0, rgb[0])
    np.testing.assert_array_equal(d0, depth[0])


def test_reader_rejects_wrong_schema(tmp_path):
    _make_recording(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps({"schema_version": 2, "intrinsics": {}}))
    with pytest.raises(ValueError):
        RecordingReader(tmp_path)


def test_publish_then_subscribe(tmp_path):
    rgb, depth = _make_recording(tmp_path, n=6)
    ep = "tcp://127.0.0.1:5578"

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 3000)
    sub.connect(ep)

    t = threading.Thread(target=publish, args=(tmp_path, ep),
                         kwargs={"fast": True, "loop": True}, daemon=True)
    t.start()
    time.sleep(0.4)
    try:
        buf = sub.recv()
    finally:
        sub.close(0)
    f = decode_frame(buf)
    assert f.device == "replay:xtion"
    assert f.rgb.shape == (8, 12, 3) and f.depth.shape == (8, 12)

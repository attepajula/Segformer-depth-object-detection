"""Publish a recorded RGB-D session on ZeroMQ — the same stream a live camera
puts out, so any schema-v1 consumer (``segformer run``) can't tell the
difference. No camera, no OpenNI2, no Rosetta: a session folder and
``numpy / pyzmq / msgpack``.
"""

from __future__ import annotations

import time
from pathlib import Path

import zmq

from stream.consumer import DEFAULT_ENDPOINT
from stream.recording import RecordingReader
from stream.wire import encode_frame


def info(path: str | Path) -> int:
    r = RecordingReader(path)
    first = last = None
    n = 0
    for h, _rgb, _d in r.frames():
        if first is None:
            first = h["timestamp_ns"]
        last = h["timestamp_ns"]
        n += 1
    dur = (last - first) / 1e9 if first is not None else 0.0
    size_mb = sum(c.stat().st_size for c in r._chunks) / 1e6
    print(f"{Path(path)}")
    print(f"  device      {r.device}")
    print(f"  frames      {n}  ({dur:.1f} s, {n / dur:.1f} fps)" if dur else f"  frames      {n}")
    print(f"  intrinsics  {r.intrinsics}")
    print(f"  on disk     {size_mb:.0f} MB in {len(r._chunks)} chunks")
    return 0


def publish(path: str | Path, endpoint: str = DEFAULT_ENDPOINT, *,
            fast: bool = False, loop: bool = False, compression: str = "none") -> int:
    reader = RecordingReader(path)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.CONFLATE, 1)     # newest-frame-wins, like the live publisher
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(endpoint)
    time.sleep(0.2)                       # let subscribers connect before frame 0

    print(f"replay {Path(path).name} -> {endpoint}  "
          f"({reader.frame_count} frames, {'fast' if fast else 'timed'}"
          f"{', loop' if loop else ''})")
    sent = 0
    try:
        while True:
            wall0 = time.monotonic()
            t0 = None
            for header, rgb, depth in reader.frames():
                if not fast:
                    if t0 is None:
                        t0 = header["timestamp_ns"]
                    target = wall0 + (header["timestamp_ns"] - t0) / 1e9
                    delay = target - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                sock.send(encode_frame(header, rgb, depth, compression), flags=zmq.NOBLOCK)
                sent += 1
            if not loop:
                break
    except KeyboardInterrupt:
        pass
    except zmq.Again:
        pass
    finally:
        sock.close(0)
    print(f"sent {sent} frames")
    return 0

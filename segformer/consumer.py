"""ZeroMQ consumer: subscribe latest-frame-wins, decode, segment.

Mirrors the capture pipeline's transport contract: SUB socket, ``CONFLATE=1``,
subscribe to ``b""``. Whatever the publisher's fps, we always process the
freshest frame and never a stale backlog.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import zmq

from segformer.frame import Frame
from segformer.wire import decode_frame

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5556"


class FrameConsumer:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, *, timeout_ms: int = 2000):
        self.endpoint = endpoint
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(endpoint)

    def recv(self) -> Frame | None:
        try:
            return decode_frame(self._sock.recv())
        except zmq.Again:
            return None

    def frames(self) -> Iterator[Frame | None]:
        """Yield frames forever; yields None on each receive timeout."""
        while True:
            yield self.recv()

    def close(self) -> None:
        self._sock.close(0)

    def __enter__(self) -> "FrameConsumer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RateMeter:
    def __init__(self, window: int = 60):
        self.window = window
        self._t: list[float] = []

    def tick(self) -> float:
        self._t.append(time.monotonic())
        if len(self._t) > self.window:
            self._t = self._t[-self.window :]
        if len(self._t) < 2 or self._t[-1] == self._t[0]:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])

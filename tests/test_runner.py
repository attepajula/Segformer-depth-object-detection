"""_Latest one-slot handoff semantics (the threading primitive in runner.py)."""

import threading
import time

from segformer.runner import _Latest


def test_get_returns_latest_only():
    s = _Latest()
    s.put(1)
    s.put(2)
    s.put(3)
    assert s.get(timeout=0.1) == 3
    assert s.get(timeout=0.05) is None  # consumed, nothing new


def test_get_blocks_until_put():
    s = _Latest()
    assert s.get(timeout=0.05) is None

    def producer():
        time.sleep(0.05)
        s.put("x")

    threading.Thread(target=producer, daemon=True).start()
    assert s.get(timeout=1.0) == "x"


def test_producer_overwrites_never_blocks():
    s = _Latest()
    for i in range(10_000):
        s.put(i)          # a slow consumer must never back this up
    assert s.get(timeout=0.1) == 9999

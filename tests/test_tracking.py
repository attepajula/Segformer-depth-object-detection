"""Tracker hysteresis: confirm after min_hits, linger for max_misses."""

import numpy as np

from segformer.segmenter import Instance, SegResult
from segformer.tracking import Tracker


def _det(present: bool, fid: int) -> SegResult:
    r = SegResult(frame_id=fid, timestamp_ns=fid)
    if present:
        m = np.zeros((20, 20), bool)
        m[4:16, 4:16] = True
        r.instances.append(Instance(0, 14, 0.9, m, (4.0, 4.0, 16.0, 16.0)))
    return r


def test_confirm_then_linger():
    tk = Tracker(min_hits=3, max_misses=6)
    # frames 1-2: seen but not yet confirmed
    assert len(tk.update(_det(True, 1)).instances) == 0
    assert len(tk.update(_det(True, 2)).instances) == 0
    # frame 3: confirmed
    assert len(tk.update(_det(True, 3)).instances) == 1
    # drop out for a few frames -> still reported (linger)
    assert len(tk.update(_det(False, 4)).instances) == 1
    assert len(tk.update(_det(False, 5)).instances) == 1
    # ... eventually gone
    for f in range(6, 12):
        tk.update(_det(False, f))
    assert len(tk.update(_det(False, 12)).instances) == 0


def test_single_frame_blip_never_reported():
    tk = Tracker(min_hits=3)
    assert len(tk.update(_det(True, 1)).instances) == 0
    for f in range(2, 10):
        assert len(tk.update(_det(False, f)).instances) == 0


def test_reset_clears():
    tk = Tracker(min_hits=2)
    tk.update(_det(True, 1))
    tk.update(_det(True, 2))
    assert tk.update(_det(True, 3)).instances
    tk.reset()
    assert len(tk.update(_det(True, 4)).instances) == 0

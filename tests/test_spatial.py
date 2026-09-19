"""Spatial-audio cue maths on synthetic frames. No model, no hardware."""

import math

import numpy as np

from consumer.spatial import Cue, CueSmoother, compute_cues, render
from segformer.segmenter import Instance, SegResult
from stream.frame import Frame, Intrinsics

H, W = 480, 640
INTR = Intrinsics(570.0, 570.0, 320.0, 240.0, W, H)


def _frame(depth_mm: np.ndarray) -> Frame:
    return Frame(schema_version=1, frame_id=1, timestamp_ns=0,
                 rgb=np.zeros((H, W, 3), np.uint8), depth=depth_mm, intrinsics=INTR,
                 depth_scale=0.001, registered=True, device="test")


def _inst(x0, x1, iid=1, label=14) -> Instance:
    m = np.zeros((H, W), bool)
    m[100:400, x0:x1] = True
    return Instance(iid, label, 0.9, m, (float(x0), 100.0, float(x1), 400.0))


def _result(*insts) -> SegResult:
    return SegResult(frame_id=1, timestamp_ns=0, instances=list(insts))


def test_door_dead_ahead_is_centred_at_its_depth():
    f = _frame(np.full((H, W), 2000, np.uint16))
    (c,) = compute_cues(f, _result(_inst(300, 340)))
    assert abs(c.azimuth_rad) < 0.03
    assert abs(c.pan) < 0.03
    assert abs(c.distance_m - 2.0) < 0.05


def test_door_to_the_right_pans_right():
    f = _frame(np.full((H, W), 2000, np.uint16))
    (c,) = compute_cues(f, _result(_inst(500, 560)))
    assert c.azimuth_rad > 0.2 and c.pan > 0.2


def test_door_to_the_left_pans_left():
    f = _frame(np.full((H, W), 2000, np.uint16))
    (c,) = compute_cues(f, _result(_inst(60, 120)))
    assert c.azimuth_rad < -0.2 and c.pan < -0.2


def test_nearer_is_louder():
    near = compute_cues(_frame(np.full((H, W), 1200, np.uint16)), _result(_inst(300, 340)))[0]
    far = compute_cues(_frame(np.full((H, W), 3500, np.uint16)), _result(_inst(300, 340)))[0]
    assert near.gain > far.gain


def test_no_depth_keeps_bearing_but_distance_unknown():
    f = _frame(np.zeros((H, W), np.uint16))
    (c,) = compute_cues(f, _result(_inst(500, 560)))
    assert c.distance_m is None
    assert c.azimuth_rad > 0.2


def test_background_bleed_on_mask_rim_does_not_drag_distance():
    depth = np.full((H, W), 6000, np.uint16)       # far wall behind
    depth[100:400, 300:340] = 2000                 # the door itself
    inst = _inst(296, 344)                         # mask rim spills 4 px onto the wall
    (c,) = compute_cues(_frame(depth), _result(inst))
    assert abs(c.distance_m - 2.0) < 0.1


def test_smoother_smooths_and_forgets():
    s = CueSmoother()
    a = Cue(1, 14, 0.0, 2.0, 0.0, 0.5)
    b = Cue(1, 14, 0.5, 4.0, math.sin(0.5), 0.25)
    s.update([a])
    (out,) = s.update([b])
    assert 0.0 < out.azimuth_rad < 0.5 and 2.0 < out.distance_m < 4.0
    for _ in range(12):
        s.update([])
    assert s._state == {}


def test_render_draws_with_and_without_cues():
    img = render([], INTR)
    assert img.dtype == np.uint8 and img.ndim == 3
    cues = [Cue(1, 14, 0.2, 2.0, 0.2, 0.5), Cue(2, 38, -0.4, None, -0.39, 0.3)]
    assert render(cues, INTR).shape == img.shape

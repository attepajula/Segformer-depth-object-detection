"""Depth door-gate on synthetic depth. No model, no hardware."""

import numpy as np

from stream.frame import Intrinsics
from segformer.geometry import backproject, door_gate

INTR = Intrinsics(570.0, 570.0, 320.0, 240.0, 640, 480)
H, W = 480, 640


def _vertical_plane(z_m: float) -> np.ndarray:
    """Fronto-parallel plane at depth z (constant), mm."""
    return np.full((H, W), int(z_m * 1000), np.uint16)


def _rect(x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_door_sized_planar_vertical_passes():
    depth = _vertical_plane(2.0)
    pts, valid = backproject(depth, INTR)
    # ~0.9 m wide, ~2.0 m tall at z=2m:  w_px = 0.9*fx/z ≈ 256 ; h_px = 2.0*570/2 ≈ 570 -> clip
    w_px = int(0.9 * INTR.fx / 2.0)
    mask = _rect(320 - w_px // 2, 0, 320 + w_px // 2, H)  # full height ≈ 1.68 m
    g = door_gate(mask, pts, valid, INTR)
    assert g.keep, g.reason


def test_wall_sized_rejected():
    depth = _vertical_plane(2.0)
    pts, valid = backproject(depth, INTR)
    g = door_gate(_rect(20, 0, 620, H), pts, valid, INTR)
    assert not g.keep
    assert g.reason in ("fills-frame", "wall-sized", "width-out-of-range")


def test_floor_rejected_as_not_vertical():
    # depth increases with row (v): a floor receding from the camera
    v = np.arange(H, dtype=np.float32)[:, None]
    depth = (800 + v * 6).astype(np.uint16).repeat(W, axis=1)
    pts, valid = backproject(depth, INTR)
    mask = _rect(240, 300, 400, 460)  # a patch of floor, door-ish bbox
    g = door_gate(mask, pts, valid, INTR)
    assert not g.keep


def test_no_depth_falls_back_to_keep():
    depth = np.zeros((H, W), np.uint16)
    pts, valid = backproject(depth, INTR)
    g = door_gate(_rect(280, 60, 360, 420), pts, valid, INTR)
    assert g.keep and g.reason in ("shallow-keep", "insufficient-depth")


def test_consolidate_merges_same_depth_blobs():
    import cv2

    from segformer.geometry import consolidate_door

    # two blobs, same depth plane -> one merged mask
    binm = np.zeros((H, W), np.uint8)
    binm[100:300, 250:330] = 1     # big
    binm[120:180, 360:400] = 1     # small, to the side
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(binm)
    cand = [c for c in range(1, nlab) if stats[c, cv2.CC_STAT_AREA] >= 150]
    assert len(cand) == 2
    depth = _vertical_plane(2.0)
    merged = consolidate_door(lab, cand, stats, depth)
    # both components present in the merged mask
    assert merged[110, 300] and merged[140, 380]


def test_consolidate_drops_far_depth_blob():
    import cv2

    from segformer.geometry import consolidate_door

    binm = np.zeros((H, W), np.uint8)
    binm[100:300, 250:330] = 1
    binm[120:200, 380:430] = 1
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(binm)
    cand = [c for c in range(1, nlab) if stats[c, cv2.CC_STAT_AREA] >= 150]
    depth = np.full((H, W), 2000, np.uint16)
    depth[:, 360:] = 5000          # second blob is 3 m further -> not the same door
    merged = consolidate_door(lab, cand, stats, depth)
    assert merged[110, 300]
    assert not merged[150, 400]

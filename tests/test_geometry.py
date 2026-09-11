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


def test_no_depth_always_kept():
    """No usable depth => defer to the RGB model, never reject on 2D shape
    alone. A close door with no depth reading can legitimately fill the frame
    or be wide -- that used to get it rejected; it no longer does."""
    depth = np.zeros((H, W), np.uint16)
    pts, valid = backproject(depth, INTR)
    small = door_gate(_rect(280, 60, 360, 420), pts, valid, INTR)
    assert small.keep and small.reason == "no-depth-keep"

    fills_frame = door_gate(_rect(0, 0, W, H), pts, valid, INTR)
    assert fills_frame.keep

    very_wide = door_gate(_rect(0, 200, W, 260), pts, valid, INTR)
    assert very_wide.keep


def test_cluster_merges_same_depth_blobs():
    import cv2

    from segformer.geometry import cluster_doors

    # two blobs, same depth plane -> one cluster, one mask
    binm = np.zeros((H, W), np.uint8)
    binm[100:300, 250:330] = 1     # big
    binm[120:180, 360:400] = 1     # small, to the side
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(binm)
    cand = [c for c in range(1, nlab) if stats[c, cv2.CC_STAT_AREA] >= 150]
    assert len(cand) == 2
    depth = _vertical_plane(2.0)
    masks = cluster_doors(lab, cand, stats, depth)
    assert len(masks) == 1
    assert masks[0][110, 300] and masks[0][140, 380]


def test_cluster_keeps_far_depth_blob_as_second_door():
    """Two blobs at different depths are two doors, not one merged mask and
    not one dropped -- this is the fix for the old 'only one door per frame'
    cap: a real second door used to be silently merged or discarded here."""
    import cv2

    from segformer.geometry import cluster_doors

    binm = np.zeros((H, W), np.uint8)
    binm[100:300, 250:330] = 1
    binm[120:200, 380:430] = 1
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(binm)
    cand = [c for c in range(1, nlab) if stats[c, cv2.CC_STAT_AREA] >= 150]
    depth = np.full((H, W), 2000, np.uint16)
    depth[:, 360:] = 5000          # second blob is 3 m further -> a different door
    masks = cluster_doors(lab, cand, stats, depth)
    assert len(masks) == 2
    near, far = (masks if masks[0][110, 300] else masks[::-1])
    assert near[110, 300] and not near[150, 400]
    assert far[150, 400] and not far[110, 300]


def test_cluster_handles_no_candidates():
    from segformer.geometry import cluster_doors

    lab = np.zeros((H, W), np.int32)
    assert cluster_doors(lab, [], np.zeros((1, 8)), np.zeros((H, W), np.uint16)) == []

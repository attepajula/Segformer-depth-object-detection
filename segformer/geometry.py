"""Depth post-process: reject RGB-only false positives using geometry.

SegFormer-B0 confuses large flat surfaces with doors. We have registered metric
depth — a real door is a *bounded, door-sized, roughly planar, vertical*
surface. A wall is planar and vertical too, but metres wide and often most of
the frame.

Camera frame: +x right, +y down, +z forward (OpenNI). "Up" is -y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segformer.frame import Intrinsics

# The mask reaching here is a high-probability door core (wall fringe already
# dropped by the prob threshold), and the door is often only partly in frame
# (short USB tether) — so height is a weak signal, width + planarity + vertical
# do the work.
DOOR_W = (0.4, 1.9)          # single leaf .. wide / double
DOOR_H_MIN = 0.9            # tolerate a cropped door
DOOR_H_MAX = 2.7
WALL_W = 2.3                 # wider => wall
NO_DEPTH_MAX_FRAC = 0.55     # without depth: reject a blob bigger than this
NO_DEPTH_MAX_ASPECT = 1.6    # without depth: reject much-wider-than-tall
MIN_VALID_FRAC = 0.25        # need this much of the mask to carry depth
PLANE_TOL = 0.06             # m
PLANE_INLIER = 0.45
MAX_NORMAL_Y = 0.55          # |n_y|: 0 vertical surface, 1 floor/ceiling

DEPTH_MERGE_TOL = 0.35       # m — components within this of the anchor's median
                            #     depth are the same door, merge before gating
MERGE_CLOSE_PX = 9          # morphological close to bridge merged sub-blobs


@dataclass
class GateResult:
    keep: bool
    reason: str
    width_m: float = 0.0
    height_m: float = 0.0
    planar: float = 0.0
    normal_y: float = 0.0
    frame_frac: float = 0.0


def backproject(depth_mm: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """(H,W) uint16 mm -> points (H,W,3) metres, valid (H,W) bool."""
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float32) / 1000.0
    valid = depth_mm > 0
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (us - intr.cx) * z / intr.fx
    y = (vs - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=-1), valid


def _fit_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n = vh[2]
    return n, float((np.abs((pts - c) @ n) < PLANE_TOL).mean())


def door_gate(mask: np.ndarray, points: np.ndarray, valid: np.ndarray,
              intr: Intrinsics) -> GateResult:
    area = int(mask.sum())
    frac = area / mask.size
    ys, xs = np.where(mask)
    bw_px = xs.max() - xs.min() + 1
    bh_px = ys.max() - ys.min() + 1

    g = GateResult(True, "ok", frame_frac=frac)

    m = mask & valid
    nvalid = int(m.sum())
    if nvalid < 80 or nvalid < MIN_VALID_FRAC * area:
        # no usable depth: judge from 2D shape only.
        if frac > NO_DEPTH_MAX_FRAC:
            return GateResult(False, "fills-frame", frame_frac=frac)
        if bw_px / max(bh_px, 1) > NO_DEPTH_MAX_ASPECT:
            return GateResult(False, "wider-than-tall", frame_frac=frac)
        return GateResult(True, "shallow-keep", frame_frac=frac)

    pts = points[m]
    z_med = float(np.median(pts[:, 2]))

    # 2. metric size from the bbox + median depth (stable when depth is sparse)
    g.width_m = bw_px * z_med / intr.fx
    g.height_m = bh_px * z_med / intr.fy

    n, inlier = _fit_plane(pts)
    g.planar, g.normal_y = inlier, float(abs(n[1]))

    if g.width_m > WALL_W:
        g.keep, g.reason = False, "wall-sized"
    elif not (DOOR_W[0] <= g.width_m <= DOOR_W[1]):
        g.keep, g.reason = False, "width-out-of-range"
    elif not (DOOR_H_MIN <= g.height_m <= DOOR_H_MAX):
        g.keep, g.reason = False, "height-out-of-range"
    elif inlier < PLANE_INLIER:
        g.keep, g.reason = False, "not-planar"
    elif g.normal_y > MAX_NORMAL_Y:
        g.keep, g.reason = False, "not-vertical"
    return g


def consolidate_door(labels: np.ndarray, cand_ids: list[int], stats: np.ndarray,
                     depth_mm: np.ndarray) -> np.ndarray:
    """Merge B0's fragmented door blobs that sit at the same depth into one mask.

    Different-sized blobs on the same depth plane are one door, not many —
    gate the union, once. ``labels``/``stats`` from connectedComponentsWithStats.
    """
    import cv2

    area = cv2.CC_STAT_AREA
    anchor = max(cand_ids, key=lambda c: stats[c, area])
    z = depth_mm.astype(np.float32) / 1000.0

    def med_depth(cid: int) -> float:
        zc = z[(labels == cid) & (depth_mm > 0)]
        return float(np.median(zc)) if zc.size > 30 else float("nan")

    z0 = med_depth(anchor)
    merged = labels == anchor
    if np.isfinite(z0):
        for c in cand_ids:
            if c == anchor:
                continue
            zc = med_depth(c)
            if np.isfinite(zc) and abs(zc - z0) <= DEPTH_MERGE_TOL:
                merged |= labels == c

    k = np.ones((MERGE_CLOSE_PX, MERGE_CLOSE_PX), np.uint8)
    merged = cv2.morphologyEx(merged.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return merged

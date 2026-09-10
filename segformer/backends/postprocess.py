"""Probability maps -> SegResult. Framework-free (numpy + cv2 only).

Both backends produce per-class probability maps the same way; everything
after that — thresholding, connected components, depth consolidation, the door
gate, and temporal smoothing — is shared here.
"""

from __future__ import annotations

import numpy as np

from stream.frame import Frame
from segformer.labels import DOORLIKE
from segformer.segmenter import Instance, SegResult

PROB_THRESH = {14: 0.85, 58: 0.85}   # door / screen-door: high, isolate the leaf
DEFAULT_PROB_THRESH = 0.45           # railing / bannister
MIN_COMP_PX = 150                    # ignore components smaller than this


def _instance(iid: int, label: int, mask: np.ndarray, pmap: np.ndarray) -> Instance:
    ys, xs = np.where(mask)
    return Instance(
        instance_id=iid,
        label=label,
        score=float(pmap[mask].mean()),
        mask=mask,
        box_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
    )


def probs_to_result(frame: Frame, prob: np.ndarray, target_ids: list[int], *,
                    min_area: int, use_depth: bool, tracker=None) -> SegResult:
    """prob: (H, W, len(target_ids)) float32 in [0, 1]."""
    import cv2

    res = SegResult(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns)
    next_id = 1
    k3 = np.ones((3, 3), np.uint8)
    points = None  # backproject lazily, once, only if a door candidate exists

    for ti, ade_id in enumerate(target_ids):
        thresh = PROB_THRESH.get(ade_id, DEFAULT_PROB_THRESH)
        binm = (prob[:, :, ti] > thresh).astype(np.uint8)
        if not binm.any():
            continue
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k3)
        nlab, lab, stats, _c = cv2.connectedComponentsWithStats(binm)
        cand = [c for c in range(1, nlab) if stats[c, cv2.CC_STAT_AREA] >= MIN_COMP_PX]
        if not cand:
            continue
        pmap = prob[:, :, ti]

        if use_depth and ade_id in DOORLIKE:
            from segformer.geometry import backproject, consolidate_door, door_gate

            if points is None:
                points = backproject(frame.depth, frame.intrinsics)
            pts, valid = points

            merged = consolidate_door(lab, cand, stats, frame.depth)
            if int(merged.sum()) < min_area:
                continue
            if not door_gate(merged, pts, valid, frame.intrinsics).keep:
                continue
            res.instances.append(_instance(next_id, ade_id, merged, pmap))
            next_id += 1
        else:
            for c in cand:
                if stats[c, cv2.CC_STAT_AREA] < min_area:
                    continue
                res.instances.append(_instance(next_id, ade_id, lab == c, pmap))
                next_id += 1

    return tracker.update(res) if tracker is not None else res

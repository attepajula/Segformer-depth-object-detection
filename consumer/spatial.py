"""MVP downstream consumer: where is each detected object, as a spatial-audio cue.

Takes a ``Frame`` (colour + depth + lens) and its detections and works out, per
object, the direction and distance a spatial-audio renderer would need.
``render`` draws what gets heard: a top-down radar plus a left/right pan and
loudness per object. The audio itself is ``consumer.audio`` — this module only
computes the cues (``Cue``, ``compute_cues``, ``CueSmoother``) that it plays.

Segformer's network never sees depth, so distance is computed here: the median
of the mask interior's valid depth readings. No reading -> distance is None
(direction is still known from the pixel column, so the cue keeps a bearing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from segformer.labels import ADE20K_TARGETS, CLASS_COLOR, CLASS_COLOR_DEFAULT
from stream.frame import Frame, Intrinsics

MIN_DEPTH_PX = 50          # fewer valid depth pixels under the mask -> distance unknown
ERODE_PX = 5               # mask rim is a blur of the 128x128 grid; median the interior
REF_DIST_M = 1.0           # loudness 1.0 at/inside this range, then 1/d
UNKNOWN_GAIN = 0.3         # loudness drawn for an object with a bearing but no distance
EMA_ALPHA = 0.4            # per-id smoothing of bearing and distance
DROP_AFTER = 10            # forget an id not seen for this many updates
MAX_RANGE_M = 5.0          # radar edge


@dataclass
class Cue:
    instance_id: int
    label: int
    azimuth_rad: float             # 0 = straight ahead, + = to the right
    distance_m: float | None       # horizontal range; None = no depth under the mask
    pan: float                     # -1 full left .. +1 full right
    gain: float                    # 0..1 loudness

    @property
    def name(self) -> str:
        return ADE20K_TARGETS.get(self.label, str(self.label))


def _gain(distance_m: float | None) -> float:
    if distance_m is None:
        return UNKNOWN_GAIN
    return float(min(1.0, REF_DIST_M / max(distance_m, 1e-3)))


def _cue(instance_id: int, label: int, az: float, dist: float | None) -> Cue:
    return Cue(instance_id, label, az, dist, float(np.sin(az)), _gain(dist))


def compute_cues(frame: Frame, result) -> list[Cue]:
    """One Cue per detection. ``result`` is a segformer ``SegResult``."""
    import cv2

    intr: Intrinsics = frame.intrinsics
    k = np.ones((ERODE_PX, ERODE_PX), np.uint8)
    cues = []
    for inst in result.instances:
        mask = inst.mask
        core = cv2.erode(mask.astype(np.uint8), k).astype(bool)
        if not core.any():
            core = mask
        valid = core & (frame.depth > 0)
        ys, xs = np.nonzero(valid)
        if xs.size >= MIN_DEPTH_PX:
            z = frame.depth[ys, xs].astype(np.float32) * frame.depth_scale
            x = (xs - intr.cx) * z / intr.fx
            mx, mz = float(np.median(x)), float(np.median(z))
            cues.append(_cue(inst.instance_id, inst.label, math.atan2(mx, mz), math.hypot(mx, mz)))
        else:
            _, cxs = np.nonzero(mask)
            az = math.atan2((float(cxs.mean()) - intr.cx) / intr.fx, 1.0)
            cues.append(_cue(inst.instance_id, inst.label, az, None))
    return cues


class CueSmoother:
    """Exponential smoothing of bearing + distance, per stable instance id."""

    def __init__(self) -> None:
        self._state: dict[int, tuple[float, float | None, int]] = {}
        self._tick = 0

    def reset(self) -> None:
        self._state.clear()

    def update(self, cues: list[Cue]) -> list[Cue]:
        self._tick += 1
        out = []
        for c in cues:
            prev = self._state.get(c.instance_id)
            az, dist = c.azimuth_rad, c.distance_m
            if prev is not None:
                az = prev[0] + EMA_ALPHA * (az - prev[0])
                if dist is not None and prev[1] is not None:
                    dist = prev[1] + EMA_ALPHA * (dist - prev[1])
            self._state[c.instance_id] = (az, dist, self._tick)
            out.append(_cue(c.instance_id, c.label, az, dist))
        for i in [i for i, s in self._state.items() if self._tick - s[2] > DROP_AFTER]:
            del self._state[i]
        return out


# -- drawing ---------------------------------------------------------------
_RADAR = 480
_PANEL = 300
_PX_PER_M = 80.0
_BG = (24, 24, 28)
_GRID = (70, 70, 78)
_TEXT = (220, 220, 220)


def _rgb(label: int) -> tuple[int, int, int]:
    return tuple(int(c) for c in CLASS_COLOR.get(label, CLASS_COLOR_DEFAULT))


def render(cues: list[Cue], intr: Intrinsics) -> np.ndarray:
    """(H, W, 3) uint8 RGB: top-down radar on the left, per-object pan/loudness on the right."""
    import cv2

    img = np.zeros((_RADAR, _RADAR + _PANEL, 3), np.uint8)
    img[:] = _BG
    ox, oy = _RADAR // 2, _RADAR - 30

    for m in range(1, int(MAX_RANGE_M) + 1):
        r = int(m * _PX_PER_M)
        cv2.ellipse(img, (ox, oy), (r, r), 0, 180, 360, _GRID, 1)
        cv2.putText(img, f"{m}m", (ox + 4, oy - r + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _GRID, 1)
    half = math.atan2(intr.width / 2, intr.fx)
    for s in (-1, 1):
        end = (int(ox + s * math.sin(half) * MAX_RANGE_M * _PX_PER_M),
               int(oy - math.cos(half) * MAX_RANGE_M * _PX_PER_M))
        cv2.line(img, (ox, oy), end, _GRID, 1)
    cv2.circle(img, (ox, oy), 5, _TEXT, -1)
    cv2.putText(img, "you", (ox - 12, oy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT, 1)

    cv2.line(img, (_RADAR, 0), (_RADAR, _RADAR), _GRID, 1)
    cv2.putText(img, "L", (_RADAR + 12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1)
    cv2.putText(img, "R", (_RADAR + _PANEL - 22, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1)

    for row, c in enumerate(sorted(cues, key=lambda c: c.distance_m if c.distance_m is not None else 1e9)):
        col = _rgb(c.label)
        dx, dy = math.sin(c.azimuth_rad), math.cos(c.azimuth_rad)
        if c.distance_m is not None:
            d = min(c.distance_m, MAX_RANGE_M)
            p = (int(ox + dx * d * _PX_PER_M), int(oy - dy * d * _PX_PER_M))
            cv2.circle(img, p, 8, col, -1)
            dist_txt = f"{c.distance_m:.1f} m"
        else:
            edge = MAX_RANGE_M * _PX_PER_M
            p = (int(ox + dx * edge), int(oy - dy * edge))
            cv2.line(img, (ox, oy), p, col, 1, cv2.LINE_AA)
            cv2.circle(img, p, 8, col, 2)
            dist_txt = "? m"
        cv2.putText(img, f"{c.name} #{c.instance_id}", (p[0] + 10, p[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        y = 50 + row * 62
        if y > _RADAR - 20:
            break
        deg = math.degrees(c.azimuth_rad)
        cv2.putText(img, f"{c.name} #{c.instance_id}  {deg:+.0f} deg  {dist_txt}",
                    (_RADAR + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        x0, x1 = _RADAR + 12, _RADAR + _PANEL - 12
        mid = (x0 + x1) // 2
        cv2.rectangle(img, (x0, y + 8), (x1, y + 20), _GRID, 1)
        # louder ear gets the bar: pan splits gain between left and right
        half_w = (x1 - x0) // 2
        left = c.gain * math.sqrt((1 - c.pan) / 2)
        right = c.gain * math.sqrt((1 + c.pan) / 2)
        cv2.rectangle(img, (mid - int(left * half_w), y + 9), (mid, y + 19), col, -1)
        cv2.rectangle(img, (mid, y + 9), (mid + int(right * half_w), y + 19), col, -1)
        cv2.putText(img, f"loudness {c.gain:.2f}" + ("  (no depth)" if c.distance_m is None else ""),
                    (x0, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _TEXT, 1)

    if not cues:
        cv2.putText(img, "nothing detected", (_RADAR + 12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT, 1)
    return img


def show(img: np.ndarray, *, title: str = "spatial") -> bool:
    """Display in its own window. Returns False when the user presses q."""
    import cv2

    cv2.imshow(title, img[:, :, ::-1])
    return (cv2.waitKey(1) & 0xFF) != ord("q")

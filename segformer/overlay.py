"""Render a SegResult over the RGB frame. OpenCV window or PNG dump."""

from __future__ import annotations

import numpy as np

from stream.frame import Frame
from segformer.labels import ADE20K_TARGETS, CLASS_COLOR, CLASS_COLOR_DEFAULT
from segformer.segmenter import SegResult


def _color(label: int) -> np.ndarray:
    return np.array(CLASS_COLOR.get(label, CLASS_COLOR_DEFAULT), dtype=np.uint8)


def render(frame: Frame, result: SegResult, *, alpha: float = 0.45, fps: float | None = None) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB image with masks + boxes drawn."""
    out = frame.rgb.copy()
    for inst in result.instances:
        col = _color(inst.label)
        m = inst.mask
        out[m] = (out[m] * (1 - alpha) + col * alpha).astype(np.uint8)

    try:
        import cv2

        for inst in result.instances:
            x0, y0, x1, y1 = (int(v) for v in inst.box_xyxy)
            col = tuple(int(c) for c in _color(inst.label))
            name = ADE20K_TARGETS.get(inst.label, str(inst.label))
            cv2.rectangle(out, (x0, y0), (x1, y1), col, 1)
            cv2.putText(out, f"{name} {inst.score:.2f}", (x0, max(y0 - 4, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        banner = f"frame {result.frame_id}  inst {len(result.instances)}"
        if fps is not None:
            banner += f"  {fps:4.1f} fps"
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(out, banner, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    except Exception:
        pass
    return out


def poll_key(delay_ms: int = 1) -> int:
    """One ``cv2.waitKey`` call. OpenCV has a single key-event queue shared by
    every window, not one per window -- call this once per loop tick and hand
    the result to every ``show()`` that iteration, or two windows' own
    ``waitKey`` calls split the same stream and each only gets some of the
    keypresses."""
    import cv2

    return cv2.waitKey(delay_ms) & 0xFF


def show(rgb: np.ndarray, *, title: str = "segformer", key: int | None = None) -> bool:
    """Display one frame. Returns False when the user asks to quit.
    ``key`` -- pass the result of one shared ``poll_key()`` when another
    window is shown the same iteration; omit it to poll here (fine when this
    is the only window)."""
    import cv2

    cv2.imshow(title, rgb[:, :, ::-1])  # RGB->BGR
    if key is None:
        key = poll_key(1)
    return key != ord("q")

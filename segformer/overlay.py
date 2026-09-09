"""Render a SegResult over the RGB frame. OpenCV window or PNG dump."""

from __future__ import annotations

import numpy as np

from segformer.frame import Frame
from segformer.model.config import ADE20K_TARGETS
from segformer.segmenter import SegResult

# fixed color per ADE20K class we surface (RGB)
_CLASS_COLOR = {
    14: (255, 130, 0),    # door — orange
    38: (60, 220, 60),    # railing — green
    95: (255, 235, 0),    # bannister — yellow
    58: (230, 60, 230),   # screen door — magenta
}
_DEFAULT = (0, 180, 255)


def _color(label: int) -> np.ndarray:
    return np.array(_CLASS_COLOR.get(label, _DEFAULT), dtype=np.uint8)


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


def show(rgb: np.ndarray, *, title: str = "segformer") -> bool:
    """Display one frame. Returns False when the user asks to quit."""
    import cv2

    cv2.imshow(title, rgb[:, :, ::-1])  # RGB->BGR
    return (cv2.waitKey(1) & 0xFF) != ord("q")

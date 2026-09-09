"""Render a SegResult over the RGB frame. OpenCV window or PNG dump."""

from __future__ import annotations

import numpy as np

from seqformer.frame import Frame
from seqformer.segmenter import SegResult

# distinct-ish colors for instance ids (RGB)
_PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
    ],
    dtype=np.uint8,
)


def _color(instance_id: int) -> np.ndarray:
    return _PALETTE[instance_id % len(_PALETTE)]


def render(frame: Frame, result: SegResult, *, alpha: float = 0.45, fps: float | None = None) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB image with masks + boxes drawn."""
    out = frame.rgb.copy()
    for inst in result.instances:
        col = _color(inst.instance_id)
        m = inst.mask
        out[m] = (out[m] * (1 - alpha) + col * alpha).astype(np.uint8)

    try:
        import cv2

        for inst in result.instances:
            x0, y0, x1, y1 = (int(v) for v in inst.box_xyxy)
            col = tuple(int(c) for c in _color(inst.instance_id))
            cv2.rectangle(out, (x0, y0), (x1, y1), col, 1)
            cv2.putText(out, f"#{inst.instance_id} {inst.score:.2f}", (x0, max(y0 - 4, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        banner = f"frame {result.frame_id}  inst {len(result.instances)}"
        if fps is not None:
            banner += f"  {fps:4.1f} fps"
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(out, banner, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    except Exception:
        pass
    return out


def show(rgb: np.ndarray, *, title: str = "seqformer") -> bool:
    """Display one frame. Returns False when the user asks to quit."""
    import cv2

    cv2.imshow(title, rgb[:, :, ::-1])  # RGB->BGR
    return (cv2.waitKey(1) & 0xFF) != ord("q")

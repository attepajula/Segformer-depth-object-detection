"""Threaded run loop: inference on a worker thread, display on the main thread.

The single-threaded loop pays, per frame: recv + decode + infer (blocks ~22 ms
on the GPU) + overlay + imshow — all serial. Here the worker does recv/decode/
infer and hands the latest (frame, result) to the main thread, which only draws.
While the main thread renders frame N, the worker is already on N+1, so overlay
+ imshow (~14 ms) hide behind the next inference instead of adding to it.

macOS requires cv2.imshow on the main thread — hence display here, not infer.
Each ZMQ socket is used by exactly one thread (the worker builds its own).
"""

from __future__ import annotations

import threading

from segformer import overlay
from stream.consumer import DEFAULT_ENDPOINT, FrameConsumer, RateMeter


class _Latest:
    """One-slot latest-wins handoff. put() overwrites; get() consumes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item = None
        self._ready = threading.Event()

    def put(self, item) -> None:
        with self._lock:
            self._item = item
        self._ready.set()

    def get(self, timeout: float):
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            self._ready.clear()
            item, self._item = self._item, None
        return item


def _show_waiting(endpoint: str) -> bool:
    """Blank 'waiting' frame so the window opens and stays quittable. False = quit."""
    import cv2
    import numpy as np

    img = np.zeros((240, 640, 3), np.uint8)
    cv2.putText(img, "waiting for frames", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    cv2.putText(img, endpoint, (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.imshow("segformer", img)
    return (cv2.waitKey(30) & 0xFF) != ord("q")


def run_threaded(endpoint: str, segmenter, *, show: bool = True,
                 save_dir=None, max_frames: int = 0, spatial: bool = False) -> int:
    if spatial:
        from consumer import spatial as spatial_view

        smoother = spatial_view.CueSmoother()
    stop = threading.Event()
    latest = _Latest()
    state = {"fps": 0.0, "n": 0}

    def worker() -> None:
        consumer = FrameConsumer(endpoint)          # socket owned by this thread
        meter = RateMeter()
        prev_id = None
        try:
            while not stop.is_set():
                frame = consumer.recv()
                if frame is None:
                    continue
                if prev_id is not None and frame.frame_id <= prev_id:
                    segmenter.reset()
                prev_id = frame.frame_id
                result, _ = segmenter.infer(frame, None)
                state["fps"] = meter.tick()
                state["n"] += 1
                latest.put((frame, result))
                if max_frames and state["n"] >= max_frames:
                    stop.set()
        finally:
            consumer.close()

    t = threading.Thread(target=worker, name="segformer-infer", daemon=True)
    t.start()

    if save_dir is not None:
        import pathlib

        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    import time

    last_frame_at = time.monotonic()
    last_warned = 0.0
    try:
        while not stop.is_set():
            item = latest.get(timeout=0.5)
            if item is None:
                now = time.monotonic()
                if now - last_frame_at > 1.0 and now - last_warned > 2.0:
                    print(f"  waiting for frames on {endpoint} ...")
                    last_warned = now
                if show and not _show_waiting(endpoint):
                    stop.set()
                continue
            last_frame_at = time.monotonic()
            frame, result = item
            img = overlay.render(frame, result, fps=state["fps"])   # infer fps
            if save_dir is not None:
                import cv2

                cv2.imwrite(str(save_dir / f"{frame.frame_id:06d}.png"), img[:, :, ::-1])
            if show and not overlay.show(img):
                stop.set()
            if spatial:
                cues = smoother.update(spatial_view.compute_cues(frame, result))
                if not spatial_view.show(spatial_view.render(cues, frame.intrinsics)):
                    stop.set()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        t.join(timeout=2.0)
        segmenter.close()
        if show or spatial:
            import cv2

            cv2.destroyAllWindows()

    print(f"processed {state['n']} frames  (infer {state['fps']:.1f} fps)")
    return state["n"]

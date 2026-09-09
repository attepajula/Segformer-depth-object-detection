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
from segformer.consumer import DEFAULT_ENDPOINT, FrameConsumer, RateMeter


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


def run_threaded(endpoint: str, segmenter, *, show: bool = True,
                 save_dir=None, max_frames: int = 0) -> int:
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

    try:
        while not stop.is_set():
            item = latest.get(timeout=0.5)
            if item is None:
                continue
            frame, result = item
            img = overlay.render(frame, result, fps=state["fps"])   # infer fps
            if save_dir is not None:
                import cv2

                cv2.imwrite(str(save_dir / f"{frame.frame_id:06d}.png"), img[:, :, ::-1])
            if show and not overlay.show(img):
                stop.set()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        t.join(timeout=2.0)
        segmenter.close()
        if show:
            import cv2

            cv2.destroyAllWindows()

    print(f"processed {state['n']} frames  (infer {state['fps']:.1f} fps)")
    return state["n"]

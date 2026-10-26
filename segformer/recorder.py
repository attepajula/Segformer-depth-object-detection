"""Composite-window video recording, with sound.

`run --record PATH` writes the overlay window and the spatial window to one
video file, side by side. Needs both windows (`--overlay`, and spatial --
on by default, don't pass `--no-spatial`), since there is nothing to put
side by side with just one.

The audio track is real, continuous, real-time -- `consumer.audio.
AudioRenderer` samples it at a fixed 44100 Hz regardless of anything else
going on. The video can't just take "one frame per inference tick" the same
way: detections arrive at the inference rate (12-30 fps, backend-dependent,
never constant), and a `cv2.VideoWriter` declared at a fixed fps assumes
its frames arrive at exactly that fixed rate. Feed it one frame per tick
and its real duration stops matching the tick count's `/fps` duration,
drifting further from the real-time audio the longer the recording runs.

`WindowRecorder` fixes this the same way `AudioRenderer` already handles
its own variable input: a dedicated thread paced to the wall clock, not to
however often new frames happen to show up. The display loop just calls
`update()` with the newest composited pair, as often as it has one; a
background thread writes a frame every `1/fps` seconds of real time,
regardless, repeating the last pair if nothing new arrived yet. The video
is then genuinely `fps` frames per real second, the same way the WAV is
genuinely 44100 samples per real second -- both tracks are real-time by
construction, so there is nothing to retime or measure afterward.

`cv2.VideoWriter` only ever writes picture -- there is no audio track.
`WindowRecorder` writes a silent video to a temp path; if audio isn't
muted, `AudioRenderer` writes its exact output samples to a temp WAV
alongside it (not a system-audio/loopback capture -- the renderer just taps
the same samples it hands to the speaker). `mux()` combines the two into
the real output path with `ffmpeg`, and cleans up the temp files either
way -- if there's no audio track or `ffmpeg` isn't installed, the output
is the video alone, and the caller is told why.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

DEFAULT_FPS = 15.0  # the video's real-time rate -- WindowRecorder's pacer
                     # thread writes one frame every 1/fps seconds of wall
                     # clock, same as AudioRenderer samples at a fixed rate


class WindowRecorder:
    """Owns a `cv2.VideoWriter`, opened and sized on the first pair given to
    `update()`. A background thread writes one frame every `1/fps` seconds
    of real time -- call `start()` once, then `update(left, right)` as
    often as a new composited pair is ready; the pacer writes whatever the
    latest one is on its own schedule, duplicating it if nothing new has
    arrived since the last tick. `write()` pads the shorter image to the
    taller one's height (black letterbox) before placing them side by
    side."""

    def __init__(self, path: str, fps: float = DEFAULT_FPS) -> None:
        self.path = path
        self.fps = fps
        self._writer = None
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, np.ndarray] | None = None
        self._pacer: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Begin the real-time pacer thread. No video is written until the
        first `update()` call gives it something to write."""
        self._pacer = threading.Thread(target=self._pacer_loop, daemon=True)
        self._pacer.start()

    def update(self, left: np.ndarray, right: np.ndarray) -> None:
        """Give the pacer thread the newest composited pair. Does not write
        immediately -- the pacer writes on its own real-time schedule, not
        on the caller's. left, right: (H, W, 3) uint8 RGB -- same
        convention as overlay.render / consumer.spatial.render."""
        with self._lock:
            self._latest = (left, right)

    def _pacer_loop(self) -> None:
        interval = 1.0 / self.fps
        next_tick = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                pair = self._latest
            if pair is not None:
                try:
                    self.write(*pair)
                except Exception as e:  # a daemon thread dying silently hides this otherwise
                    print(f"  --record: video writer failed ({e}) -- stopping")
                    self._stop.set()
                    break
            next_tick += interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop.wait(sleep_for)   # a wait(), not sleep() -- close() can cut it short
            else:
                next_tick = time.monotonic()  # fell behind -- resync, don't burst-catch-up

    def write(self, left: np.ndarray, right: np.ndarray) -> None:
        """Write one frame now, bypassing the pacer -- what the pacer calls
        internally; tests call it directly to check the compositing without
        a background thread in the way."""
        import cv2

        h = max(left.shape[0], right.shape[0])
        frame = np.hstack([_pad_to_height(left, h), _pad_to_height(right, h)])
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h_out, w_out = frame.shape[:2]
            self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w_out, h_out))
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open video writer for {self.path!r}")
        self._writer.write(frame[:, :, ::-1])  # RGB -> BGR for cv2

    def close(self) -> None:
        self._stop.set()
        if self._pacer is not None:
            self._pacer.join(timeout=2.0)
            self._pacer = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    pad = np.zeros((h - img.shape[0], img.shape[1], img.shape[2]), img.dtype)
    return np.vstack([img, pad])


# WAV header is 44 bytes; a file at or below that has no actual audio frames
# in it (e.g. the audio device never opened, so the writer thread ran but the
# callback never fed it anything) -- treat that the same as "no audio".
_WAV_HEADER_BYTES = 44


def temp_paths(record_path: str) -> tuple[str, str]:
    """Where `WindowRecorder` and `AudioRenderer(record_wav=...)` write
    while a run is in progress -- `(video_path, audio_path)`, both derived
    from the real `--record` path so they're easy to spot if `mux()` ever
    leaves them behind (e.g. the process was killed before it could run).

    The temp video is always `.mp4`, regardless of `record_path`'s own
    extension: this file is `mux()`'s input, never shown to the user, and
    `cv2.VideoWriter` picks its container from the extension it's given --
    an unusual one (an unfinished `--record out/demo.mp`, say) makes it
    fail to open, for no reason the final filename should decide."""
    base, _ext = os.path.splitext(record_path)
    return f"{base}.video.mp4", f"{base}.audio.wav"


def mux(video_path: str, audio_path: str | None, out_path: str) -> bool:
    """Combine a silent video and a WAV track into `out_path` with `ffmpeg`.
    Always leaves `video_path`/`audio_path` cleaned up. Returns True if the
    output actually has sound; False means `out_path` is the video alone
    (no audio track, `ffmpeg` missing, the video was never actually
    written, or the mux itself failed) -- the caller should tell the user
    which."""
    import shutil
    import subprocess

    if not os.path.exists(video_path):
        # write() never opened the writer (or every frame failed before
        # this point) -- nothing to mux or fall back to.
        for p in (video_path, audio_path):
            if p and os.path.exists(p):
                os.remove(p)
        return False

    have_audio = (audio_path is not None and os.path.exists(audio_path)
                 and os.path.getsize(audio_path) > _WAV_HEADER_BYTES)
    have_ffmpeg = shutil.which("ffmpeg") is not None

    ok = False
    if have_audio and have_ffmpeg:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                 "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
                check=True, capture_output=True, text=True,
            )
            ok = True
        except subprocess.CalledProcessError:
            ok = False

    if not ok:
        os.replace(video_path, out_path)

    for p in (video_path, audio_path):
        if p and os.path.exists(p):
            os.remove(p)

    return ok

"""Spatial audio for the detections `consumer.spatial` already draws.

One continuous tone per sound class — door, railing, stairs (see
``segformer.labels.SOUND_CLASS`` for the id -> class grouping: both door
types share a tone, same for railing/bannister). When several instances of
a class are in frame, the nearest (loudest) one drives that class's pan and
level for this update. Each voice is panned with both amplitude (ILD) and a
small interaural delay (ITD) -- amplitude alone reads as "quieter on one
side", the delay is what makes it read as "coming from there".

Mixing is pure numpy (``_render_block``), so it's unit-testable without
audio hardware. ``AudioRenderer`` wraps that in a ``sounddevice`` output
stream and is the only part that touches hardware; importing this module
never requires sounddevice to be installed, only calling ``start()`` does.
"""

from __future__ import annotations

import threading

import numpy as np

from segformer.labels import SOUND_CLASS

SAMPLE_RATE = 44100
BLOCKSIZE = 2048           # frames/callback -- big enough that Python+numpy never misses
                           # the deadline (small default blocks -> xruns -> crackle)
RAMP_S = 0.05              # click-free fade toward a new target gain/pan
MASTER_GAIN = 0.5          # headroom before the soft clip, so 2-3 voices at once don't distort
MAX_ITD_S = 0.0007         # interaural time delay at hard pan (~0.7 ms, human head-width
                           # range). Amplitude panning alone (ILD) reads as "quieter on one
                           # side", not "coming from there" -- a real arrival-time offset
                           # between ears is what actually does that.
MAX_ITD_SAMPLES = MAX_ITD_S * SAMPLE_RATE
PAN_EXAGGERATION = 2.2     # Cue.pan = sin(azimuth), true to the geometry -- but a true pan
                           # barely moves for a few degrees off-centre and the effect reads
                           # as flat. Boost it (then clamp) so a modest angle still swings
                           # audibly toward an ear; the radar view is untouched, this is
                           # audio-only exaggeration.

# voice name -> tone Hz. Whole numbers so a cycle divides SAMPLE_RATE evenly
# -- lets the phase index wrap every second with no discontinuity (see
# AudioRenderer._sample_idx). No tremolo/pulsing: it read as drumming, not a
# usable cue -- each class is just its own pitch.
VOICES: dict[str, int] = {
    "door": 220,
    "railing": 440,
    "stairs": 660,
}


def voice_for(label: int) -> str | None:
    """Sound class for an ADE20K label id, or None if it doesn't get one."""
    return SOUND_CLASS.get(label)


def levels_for_cues(cues) -> dict[str, tuple[float, float]]:
    """Reduce a frame's Cues to one (gain, pan) target per voice -- the
    nearest instance of each sound class. A class absent from ``cues`` is
    left out (caller treats that as gain 0)."""
    best: dict[str, tuple[float, float]] = {}
    best_gain: dict[str, float] = {}
    for c in cues:
        voice = voice_for(c.label)
        if voice is None:
            continue
        if voice not in best_gain or c.gain > best_gain[voice]:
            best_gain[voice] = c.gain
            pan = max(-1.0, min(1.0, c.pan * PAN_EXAGGERATION))
            best[voice] = (c.gain, pan)
    return best


def _render_block(frames: int, sample_idx: int,
                   levels: dict[str, tuple[float, float]]) -> np.ndarray:
    """(frames, 2) float32 stereo block. ``levels``: voice -> (gain, pan),
    already ramped by the caller -- this function is just the synthesis."""
    n = np.arange(frames, dtype=np.float64)
    mix = np.zeros((frames, 2), np.float64)
    for voice, (gain, pan) in levels.items():
        if gain <= 0:
            continue
        freq = VOICES[voice]
        # ITD: the near ear hears it first, the far ear is delayed -- pan > 0
        # (right) delays the left channel, pan < 0 delays the right.
        delay = MAX_ITD_SAMPLES * abs(pan)
        delay_l, delay_r = (delay, 0.0) if pan >= 0 else (0.0, delay)
        t_l = (sample_idx - delay_l + n) / SAMPLE_RATE
        t_r = (sample_idx - delay_r + n) / SAMPLE_RATE
        left = gain * np.sqrt(max(0.0, (1 - pan) / 2))
        right = gain * np.sqrt(max(0.0, (1 + pan) / 2))
        mix[:, 0] += left * np.sin(2.0 * np.pi * freq * t_l)
        mix[:, 1] += right * np.sin(2.0 * np.pi * freq * t_r)
    return np.tanh(mix * MASTER_GAIN).astype(np.float32)


class AudioRenderer:
    """Owns the output stream. ``update()`` from the frame loop (any
    thread); the PortAudio callback runs on its own thread and only reads
    the last targets under a lock -- never blocks on frame processing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._targets: dict[str, tuple[float, float]] = {}
        self._cur: dict[str, list[float]] = {v: [0.0, 0.0] for v in VOICES}
        self._sample_idx = 0
        self._stream = None

    def start(self) -> bool:
        """Open the output stream. Returns False (and warns once) if
        ``sounddevice`` / the audio device isn't available -- the caller
        keeps running silently, visual-only."""
        try:
            import sounddevice as sd
        except ImportError:
            print("  spatial audio: `sounddevice` not installed (pip install "
                  "'segformer[audio]') -- running silent")
            return False
        try:
            self._stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2,
                                           dtype="float32", blocksize=BLOCKSIZE,
                                           callback=self._callback)
            self._stream.start()
            return True
        except Exception as e:  # no output device, PortAudio init failure, ...
            print(f"  spatial audio: couldn't open an output device ({e}) -- running silent")
            self._stream = None
            return False

    def update(self, cues) -> None:
        targets = levels_for_cues(cues)
        with self._lock:
            self._targets = targets

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, outdata, frames, _time_info, _status) -> None:  # pragma: no cover (needs hardware)
        try:
            with self._lock:
                targets = self._targets
            alpha = min(1.0, (frames / SAMPLE_RATE) / RAMP_S)
            levels = {}
            for voice, cur in self._cur.items():
                tgain, tpan = targets.get(voice, (0.0, cur[1]))
                cur[0] += alpha * (tgain - cur[0])
                cur[1] += alpha * (tpan - cur[1])
                levels[voice] = (cur[0], cur[1])
            outdata[:] = _render_block(frames, self._sample_idx, levels)
            self._sample_idx = (self._sample_idx + frames) % SAMPLE_RATE
        except Exception as e:
            outdata.fill(0.0)
            print(f"  spatial audio callback error: {e}")

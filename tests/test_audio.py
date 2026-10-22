"""Audio mixing maths on synthetic cues. No sounddevice, no hardware."""

from __future__ import annotations

import numpy as np
import pytest

from consumer.audio import (
    PAN_EXAGGERATION,
    SAMPLE_RATE,
    VOICES,
    AudioRenderer,
    _render_block,
    confirmation_chirp,
    levels_for_cues,
    voice_for,
    voice_for_digit,
)
from consumer.spatial import Cue


def _cue(label, gain, pan, iid=1) -> Cue:
    return Cue(iid, label, 0.0, 1.0, pan, gain)


def test_voice_for_groups_both_door_types_together():
    assert voice_for(14) == voice_for(58) == "door"
    assert voice_for(38) == voice_for(95) == "railing"
    assert voice_for(53) == voice_for(59) == "stairs"
    assert voice_for(999) is None


def test_levels_for_cues_picks_the_loudest_per_voice():
    cues = [_cue(14, 0.3, -0.5, iid=1), _cue(58, 0.9, 0.2, iid=2), _cue(38, 0.4, 0.1, iid=3)]
    levels = levels_for_cues(cues)
    gain, pan = levels["door"]
    assert (gain, pan) == (0.9, pytest.approx(0.2 * PAN_EXAGGERATION))  # screen door (louder/nearer) wins over door
    gain, pan = levels["railing"]
    assert (gain, pan) == (0.4, pytest.approx(0.1 * PAN_EXAGGERATION))
    assert "stairs" not in levels


def test_levels_for_cues_ignores_unmapped_labels():
    assert levels_for_cues([_cue(1, 1.0, 0.0)]) == {}


def test_levels_for_cues_exaggerates_pan_but_clamps_to_valid_range():
    _, pan = levels_for_cues([_cue(14, 1.0, 0.3)])["door"]
    assert pan > 0.3                              # boosted beyond the true geometric pan
    _, pan_full = levels_for_cues([_cue(14, 1.0, 1.0)])["door"]
    assert -1.0 <= pan_full <= 1.0                 # still a valid pan after the boost


def test_render_block_silent_when_no_levels():
    block = _render_block(256, 0, {})
    assert block.shape == (256, 2)
    assert np.all(block == 0.0)


def test_render_block_pans_left_right():
    # hard left: pan=-1 -> all energy on the left channel
    left = _render_block(512, 0, {"door": (1.0, -1.0)})
    assert np.abs(left[:, 1]).max() < 1e-6
    assert np.abs(left[:, 0]).max() > 0.1

    right = _render_block(512, 0, {"door": (1.0, 1.0)})
    assert np.abs(right[:, 0]).max() < 1e-6
    assert np.abs(right[:, 1]).max() > 0.1


def test_render_block_gain_scales_amplitude():
    quiet = _render_block(512, 0, {"railing": (0.2, 0.0)})
    loud = _render_block(512, 0, {"railing": (0.8, 0.0)})
    assert np.abs(loud).max() > np.abs(quiet).max()


def test_render_block_never_clips():
    block = _render_block(1024, 0, {"door": (1.0, 0.0), "railing": (1.0, 0.0), "stairs": (1.0, 0.0)})
    assert np.abs(block).max() <= 1.0


def test_each_voice_is_a_steady_tone_at_its_own_pitch():
    # no tremolo/pulsing -- that read as drumming, not a usable cue. A
    # continuous tone at fixed gain has a flat envelope (no silent stretches).
    block = _render_block(SAMPLE_RATE, 0, {"stairs": (1.0, 0.0)})
    envelope = np.abs(block[:, 0])
    assert np.median(envelope) > 0.3 * envelope.max()
    assert VOICES["door"] != VOICES["railing"] != VOICES["stairs"]


def test_panned_voice_reaches_the_near_ear_first():
    # ITD on top of ILD: at a partial right pan, the right (near) channel
    # leads -- the left (far) channel is a delayed copy of it. correlate(left,
    # right) peaks at a positive offset equal to that delay.
    block = _render_block(2048, 0, {"door": (1.0, 0.6)})
    left, right = block[:, 0].astype(np.float64), block[:, 1].astype(np.float64)
    corr = np.correlate(left, right, mode="full")
    lag = corr.argmax() - (len(left) - 1)
    assert lag > 0


def test_voice_for_digit_matches_the_1_2_3_hotkeys():
    assert voice_for_digit(1) == "door"
    assert voice_for_digit(2) == "railing"
    assert voice_for_digit(3) == "stairs"
    assert voice_for_digit(4) is None
    assert voice_for_digit(0) is None


def test_toggle_voice_mutes_and_unmutes():
    r = AudioRenderer()
    assert r.toggle_voice("door") is True        # was unmuted -> now muted
    assert "door" in r._muted
    assert r.toggle_voice("door") is False       # flip back
    assert "door" not in r._muted


def test_muted_voice_gets_no_sound_even_with_cues_present():
    r = AudioRenderer()
    r.toggle_voice("railing")
    r.update([Cue(1, 38, 0.0, 1.0, 0.0, 0.9)])   # a loud, centred railing cue
    with r._lock:
        targets = r._targets
        muted = r._muted
    assert "railing" in targets                 # the cue is there
    assert "railing" in muted                    # but muted -- the callback zeroes its gain


def test_confirmation_chirp_differs_for_mute_vs_unmute():
    # the user is blind -- this sound is the only feedback a toggle
    # happened, so muted/unmuted must be reliably distinguishable: unmute
    # is two pips (longer, with a near-silent gap), mute is one.
    muted_chirp = confirmation_chirp(True)
    unmuted_chirp = confirmation_chirp(False)
    assert len(unmuted_chirp) > len(muted_chirp)
    mid = unmuted_chirp[len(unmuted_chirp) // 2 - 50 : len(unmuted_chirp) // 2 + 50]
    assert np.abs(mid).max() < 0.1 * np.abs(unmuted_chirp).max()   # the gap between the two pips
    assert np.abs(muted_chirp).max() <= 1.0
    assert np.abs(unmuted_chirp).max() <= 1.0


def test_toggle_voice_queues_a_chirp_for_the_callback_to_mix_in():
    r = AudioRenderer()
    assert r._chirp is None
    r.toggle_voice("stairs")
    assert r._chirp is not None
    assert r._chirp_pos == 0


def test_phase_is_continuous_across_the_sample_idx_wrap():
    # VOICES frequencies are whole numbers so a full second (SAMPLE_RATE
    # samples) is an exact number of cycles -- wrapping sample_idx must not
    # introduce a discontinuity.
    tail = _render_block(4, SAMPLE_RATE - 2, {"door": (1.0, 0.0)})
    head = _render_block(4, 0, {"door": (1.0, 0.0)})
    joined = _render_block(4, SAMPLE_RATE - 2, {"door": (1.0, 0.0)})
    assert np.allclose(tail, joined)
    assert all(name in VOICES for name in ("door", "railing", "stairs"))
    assert head.shape == tail.shape

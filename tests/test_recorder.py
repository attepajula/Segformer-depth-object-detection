"""WindowRecorder: compositing two windows into one video frame. The actual
cv2.VideoWriter encode/write is exercised against a real temp file -- cv2
writes video locally without needing display hardware."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from segformer.recorder import WindowRecorder, _pad_to_height, mux, temp_paths


def test_pad_to_height_leaves_already_tall_image_alone():
    img = np.zeros((480, 640, 3), np.uint8)
    out = _pad_to_height(img, 480)
    assert out is img


def test_pad_to_height_adds_black_rows_below():
    img = np.full((400, 100, 3), 200, np.uint8)
    out = _pad_to_height(img, 480)
    assert out.shape == (480, 100, 3)
    assert np.all(out[:400] == 200)
    assert np.all(out[400:] == 0)


def test_write_composites_side_by_side(tmp_path):
    cv2 = pytest.importorskip("cv2")

    left = np.full((480, 640, 3), 50, np.uint8)
    right = np.full((480, 300, 3), 150, np.uint8)
    out = tmp_path / "demo.mp4"

    rec = WindowRecorder(str(out), fps=10.0)
    try:
        for _ in range(3):
            rec.write(left, right)
    finally:
        rec.close()

    assert out.exists()
    assert out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok
        assert frame.shape[:2] == (480, 940)  # 640 + 300, same height
    finally:
        cap.release()


def test_write_pads_mismatched_heights(tmp_path):
    pytest.importorskip("cv2")

    left = np.full((480, 640, 3), 50, np.uint8)
    right = np.full((300, 300, 3), 150, np.uint8)  # shorter than left
    out = tmp_path / "demo.mp4"

    rec = WindowRecorder(str(out), fps=10.0)
    try:
        rec.write(left, right)
    finally:
        rec.close()

    assert out.exists() and out.stat().st_size > 0


def test_temp_paths_derive_from_the_record_path():
    video, audio = temp_paths("out/demo.mp4")
    assert video == "out/demo.video.mp4"
    assert audio == "out/demo.audio.wav"


def test_temp_paths_ignore_an_unusual_output_extension():
    # a typo'd/truncated --record path (e.g. "out/demo.mp") must not become
    # the temp video's own extension: cv2.VideoWriter picks its container
    # from that extension and fails to open on an unrecognised one.
    video, audio = temp_paths("out/demo.mp")
    assert video == "out/demo.video.mp4"
    assert audio == "out/demo.audio.wav"


def _tiny_wav(path, seconds=0.2) -> None:
    n = int(44100 * seconds)
    pcm = np.zeros((n, 2), "<i2").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(pcm)


def test_mux_falls_back_to_video_only_with_no_audio(tmp_path):
    pytest.importorskip("cv2")

    video = tmp_path / "demo.video.mp4"
    out = tmp_path / "demo.mp4"
    rec = WindowRecorder(str(video), fps=10.0)
    rec.write(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8))
    rec.close()

    has_sound = mux(str(video), None, str(out))

    assert has_sound is False
    assert out.exists()
    assert not video.exists()        # consumed (renamed into place)


def test_mux_falls_back_when_wav_has_no_audio_frames(tmp_path):
    pytest.importorskip("cv2")

    video = tmp_path / "demo.video.mp4"
    audio = tmp_path / "demo.audio.wav"
    out = tmp_path / "demo.mp4"
    rec = WindowRecorder(str(video), fps=10.0)
    rec.write(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8))
    rec.close()
    _tiny_wav(audio, seconds=0.0)    # header only, no frames

    has_sound = mux(str(video), str(audio), str(out))

    assert has_sound is False
    assert out.exists()
    assert not video.exists()
    assert not audio.exists()        # cleaned up even though unused


def test_mux_is_defensive_when_the_video_was_never_written(tmp_path):
    # the original crash shape: write() raised before any frame was ever
    # actually encoded (bad extension, disk full, ...), so the temp video
    # file doesn't exist. mux() must say so, not crash the whole run.
    missing_video = tmp_path / "demo.video.mp4"
    out = tmp_path / "demo.mp4"

    has_sound = mux(str(missing_video), None, str(out))

    assert has_sound is False
    assert not out.exists()


def test_pacer_writes_nothing_before_the_first_update(tmp_path):
    out = tmp_path / "demo.mp4"
    rec = WindowRecorder(str(out), fps=20.0)
    rec.start()
    try:
        import time

        time.sleep(0.1)   # a couple of pacer ticks, but never given a pair
    finally:
        rec.close()

    assert not out.exists()


def test_pacer_duplicates_the_latest_pair_at_a_fixed_real_rate(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import time

    out = tmp_path / "demo.mp4"
    rec = WindowRecorder(str(out), fps=20.0)   # one tick every 0.05 s
    rec.start()
    try:
        rec.update(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8))
        time.sleep(0.32)   # ~6-7 ticks' worth of real time, one update() call
    finally:
        rec.close()

    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    assert n >= 4   # genuinely paced in real time, not "one update = one frame"


def test_pacer_writes_the_newest_pair_each_tick(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import time

    out = tmp_path / "demo.mp4"
    rec = WindowRecorder(str(out), fps=20.0)
    rec.start()
    try:
        rec.update(np.full((10, 10, 3), 10, np.uint8), np.full((10, 10, 3), 10, np.uint8))
        time.sleep(0.12)
        rec.update(np.full((10, 10, 3), 250, np.uint8), np.full((10, 10, 3), 250, np.uint8))
        time.sleep(0.12)
    finally:
        rec.close()

    cap = cv2.VideoCapture(str(out))
    try:
        last = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
    finally:
        cap.release()

    assert last is not None
    assert last.mean() > 200   # the second, brighter pair -- not stuck on the first


def test_mux_combines_video_and_audio_with_ffmpeg(tmp_path):
    import shutil

    pytest.importorskip("cv2")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    video = tmp_path / "demo.video.mp4"
    audio = tmp_path / "demo.audio.wav"
    out = tmp_path / "demo.mp4"
    rec = WindowRecorder(str(video), fps=10.0)
    for _ in range(3):
        rec.write(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8))
    rec.close()
    _tiny_wav(audio, seconds=0.3)

    has_sound = mux(str(video), str(audio), str(out))

    assert has_sound is True
    assert out.exists() and out.stat().st_size > 0
    assert not video.exists() and not audio.exists()

"""Model smoke test. Needs the mlx extra + cached HF weights (~15 MB).
Skips cleanly offline / without weights."""

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")
pytest.importorskip("safetensors")


@pytest.fixture(scope="module")
def model():
    from segformer.model import load_segformer

    try:
        return load_segformer("b0")
    except Exception as e:  # network down, no cache
        pytest.skip(f"weights unavailable: {e}")


def test_segment_output_contract(model):
    rgb = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    lab = model.segment(rgb)
    assert lab.shape == (480, 640)
    assert lab.dtype == np.int32
    assert lab.min() >= 0 and lab.max() < 150


def test_logits_shape(model):
    rgb = np.zeros((240, 320, 3), np.uint8)
    lg = model.logits(rgb)
    assert tuple(lg.shape) == (240, 320, 150)


def test_mlx_backend_returns_target_instances(model):
    from segformer.backends import make_segmenter
    from segformer.frame import Frame, Intrinsics
    from segformer.model.config import ADE20K_TARGETS

    z = None
    import pathlib

    rec = pathlib.Path("/Users/porpyla/projut/xtion/session1/chunk_00003.npz")
    if not rec.exists():
        pytest.skip("no session1 recording to segment")
    z = np.load(rec)
    rgb, depth = z["rgb"][5], z["depth"][5]
    fr = Frame(1, 0, 0, rgb, depth, Intrinsics(570, 570, 320, 240, 640, 480),
              0.001, True, "test")

    seg = make_segmenter("mlx")
    seg.load()
    res, _ = seg.infer(fr, None)
    for inst in res.instances:
        assert inst.label in ADE20K_TARGETS
        assert 0.0 <= inst.score <= 1.0
        assert inst.mask.shape == (480, 640)

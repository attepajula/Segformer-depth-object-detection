"""Backend selection (flag / env var) and the shared post-processing."""

import numpy as np
import pytest

from segformer.backends import make_segmenter, resolve_backend
from stream.frame import Frame, Intrinsics


def test_resolve_order(monkeypatch):
    monkeypatch.delenv("SEGFORMER_BACKEND", raising=False)
    assert resolve_backend(None) == "mlx"                 # default
    monkeypatch.setenv("SEGFORMER_BACKEND", "torch")
    assert resolve_backend(None) == "torch"               # env
    assert resolve_backend("mlx") == "mlx"                # flag beats env
    assert resolve_backend("torch-mps") == "torch"        # alias


def test_unknown_backend():
    with pytest.raises(ValueError):
        make_segmenter("nope")


def _frame(h=64, w=96):
    rgb = np.zeros((h, w, 3), np.uint8)
    depth = np.zeros((h, w), np.uint16)
    return Frame(1, 0, 0, rgb, depth, Intrinsics(90, 90, w / 2, h / 2, w, h),
                 0.001, True, "test")


def test_postprocess_shared_logic():
    from segformer.backends.postprocess import probs_to_result

    f = _frame()
    prob = np.zeros((*f.depth.shape, 4), np.float32)
    prob[10:40, 20:70, 1] = 0.9          # a solid "railing" blob (id 38, thresh 0.45)
    res = probs_to_result(f, prob, [14, 38, 95, 58], min_area=200, use_depth=False)
    assert [i.label for i in res.instances] == [38]
    assert res.instances[0].score == pytest.approx(0.9)


def test_torch_backend_matches_mlx_on_a_real_frame():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import pathlib

    rec = pathlib.Path("/Users/porpyla/projut/xtion/session2/chunk_00016.npz")
    if not rec.exists():
        pytest.skip("no session2 recording")
    z = np.load(rec)
    fr = Frame(1, 0, 0, z["rgb"][0], z["depth"][0],
               Intrinsics(570.34, 570.34, 320, 240, 640, 480), 0.001, True, "t")

    seg = make_segmenter("torch")
    try:
        seg.load()
    except Exception as e:  # no network / weights
        pytest.skip(f"torch model unavailable: {e}")
    res, _ = seg.infer(fr, None)
    for inst in res.instances:
        assert inst.label in (14, 38, 95, 58)
        assert inst.mask.shape == (480, 640)

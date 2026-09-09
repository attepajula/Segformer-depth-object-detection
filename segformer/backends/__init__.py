"""Segmenter implementations. Import lazily — torch and mlx are optional."""

from __future__ import annotations

from segformer.segmenter import Segmenter


def make_segmenter(name: str, **kwargs) -> Segmenter:
    name = name.lower()
    if name in ("torch", "torch-mps", "mps"):
        from segformer.backends.torch_mps import TorchMpsSegmenter

        return TorchMpsSegmenter(**kwargs)
    if name == "mlx":
        from segformer.backends.mlx_backend import MlxSegmenter

        return MlxSegmenter(**kwargs)
    if name == "dummy":
        from segformer.backends.dummy import DummySegmenter

        return DummySegmenter(**kwargs)
    raise ValueError(f"unknown segmenter backend {name!r} (torch-mps|mlx|dummy)")


__all__ = ["Segmenter", "make_segmenter"]

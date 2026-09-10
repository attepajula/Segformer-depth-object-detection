"""Segmenter implementations. Import lazily — torch and mlx are optional.

Backend selection order (highest first):
  1. explicit name passed here / the ``--backend`` flag
  2. ``$SEGFORMER_BACKEND``
  3. ``"mlx"``
"""

from __future__ import annotations

import os

from segformer.segmenter import Segmenter

_ALIASES = {"torch-mps": "torch", "mps": "torch"}
DEFAULT_BACKEND = "mlx"


def resolve_backend(name: str | None) -> str:
    chosen = name or os.environ.get("SEGFORMER_BACKEND") or DEFAULT_BACKEND
    return _ALIASES.get(chosen.lower(), chosen.lower())


def make_segmenter(name: str | None = None, **kwargs) -> Segmenter:
    backend = resolve_backend(name)
    if backend == "mlx":
        from segformer.backends.mlx_backend import MlxSegmenter

        return MlxSegmenter(**kwargs)
    if backend == "torch":
        from segformer.backends.torch_backend import TorchSegmenter

        return TorchSegmenter(**kwargs)
    if backend == "dummy":
        from segformer.backends.dummy import DummySegmenter

        return DummySegmenter(**kwargs)
    raise ValueError(f"unknown segmenter backend {backend!r} (mlx | torch | dummy)")


__all__ = ["Segmenter", "make_segmenter", "resolve_backend", "DEFAULT_BACKEND"]

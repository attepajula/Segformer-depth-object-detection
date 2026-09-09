"""Build a SegFormer and load MLX-ready weights via the registry.

``load_model()`` — no HF, no transformers, no key remap at runtime. The registry
resolves the model name to a local ``model.mlx.safetensors`` (downloading +
converting on first use), and this just reads it.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from segformer.model.config import CONFIGS
from segformer.model.registry import ModelSpec, artifact_path, get, resolve
from segformer.model.segformer import SegFormer


def _load_into(model: SegFormer, art: Path) -> SegFormer:
    model.load_weights(list(mx.load(str(art)).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def load_model(name: str | None = None, *, checkpoint: str | None = None) -> SegFormer:
    """Load a registry model by name (default from models.toml).

    ``checkpoint`` overrides with a local MLX-ready safetensors (as produced by
    ``segformer models convert``); its arch is taken from ``name``'s spec, else
    the default model's.
    """
    spec: ModelSpec = get(name)
    model = SegFormer(CONFIGS[spec.arch])
    if checkpoint:
        return _load_into(model, Path(checkpoint))
    art, _ = resolve(name)
    return _load_into(model, art)

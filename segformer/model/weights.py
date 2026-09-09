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


def _load_into(model: SegFormer, art: Path, *, fp16: bool) -> SegFormer:
    model.load_weights(list(mx.load(str(art)).items()), strict=True)
    if fp16:
        # matmul/conv weights to fp16; keep norms in fp32 for stability.
        import mlx.nn as nn

        model.set_dtype(mx.float16)
        for _, m in model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm)):
                m.set_dtype(mx.float32)
    model.eval()
    mx.eval(model.parameters())
    return model


def load_model(name: str | None = None, *, checkpoint: str | None = None,
               fp16: bool = True) -> SegFormer:
    """Load a registry model by name (default from models.toml).

    ``checkpoint`` overrides with a local MLX-ready safetensors (as produced by
    ``segformer models convert``). ``fp16`` casts weights to float16 (norms stay
    fp32) — ~2x faster forward, negligible accuracy change for this model.
    """
    spec: ModelSpec = get(name)
    model = SegFormer(CONFIGS[spec.arch])
    art = Path(checkpoint) if checkpoint else resolve(name)[0]
    return _load_into(model, art, fp16=fp16)

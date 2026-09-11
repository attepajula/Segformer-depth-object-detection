"""Convert a HF SegFormer safetensors file to an MLX-ready one.

Runtime never calls this — ``segformer models convert`` does, once, producing
``model.mlx.safetensors`` (keys already remapped, conv kernels already
NCHW->NHWC). Loading that artifact needs no HF/transformers code.
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx

from segformer.model.config import CONFIGS
from segformer.model.segformer import SegFormer


def remap_key(k: str) -> str | None:
    """HF SegformerForSemanticSegmentation key -> this package's module path.
    None => drop the tensor."""
    if k.endswith("num_batches_tracked"):
        return None
    k = k.replace("segformer.encoder.", "encoder.")
    k = re.sub(r"^encoder\.block\.", "encoder.stages.", k)
    k = k.replace("encoder.layer_norm.", "encoder.stage_norms.")
    k = k.replace(".attention.self.query.", ".attn.query.")
    k = k.replace(".attention.self.key.", ".attn.key.")
    k = k.replace(".attention.self.value.", ".attn.value.")
    k = k.replace(".attention.self.sr.", ".attn.sr.")
    k = k.replace(".attention.self.layer_norm.", ".attn.sr_norm.")
    k = k.replace(".attention.output.dense.", ".attn.proj.")
    k = k.replace(".mlp.dwconv.dwconv.", ".mlp.dwconv.")
    k = re.sub(r"decode_head\.linear_c\.(\d)\.proj\.", r"decode_head.linear_c.\1.", k)
    return k


def _load_raw(src: Path) -> dict[str, mx.array]:
    """Read a HF checkpoint (.safetensors or torch .bin) into MLX arrays."""
    if src.suffix == ".bin":
        import torch

        state = torch.load(str(src), map_location="cpu", weights_only=True)
        return {k: mx.array(v.float().numpy()) for k, v in state.items()}
    return mx.load(str(src))


def convert(src: str | Path, arch: str) -> dict[str, mx.array]:
    """Read a HF checkpoint, return the MLX param dict. Raises on key mismatch."""
    raw = _load_raw(Path(src))
    out: dict[str, mx.array] = {}
    for hf_key, arr in raw.items():
        mlx_key = remap_key(hf_key)
        if mlx_key is None:
            continue
        if arr.ndim == 4:  # conv kernel (out, in, kH, kW) -> (out, kH, kW, in)
            arr = mx.transpose(arr, (0, 2, 3, 1))
        out[mlx_key] = arr

    # parity check against a fresh model's parameter names
    ref = SegFormer(CONFIGS[arch])
    want = {k for k, _ in _flatten(ref.parameters())}
    got = set(out)
    missing, extra = want - got, got - want
    if missing or extra:
        raise ValueError(f"key mismatch: missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}")
    return out


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flatten(v, f"{prefix}{i}.")
    else:
        yield prefix.rstrip("."), tree

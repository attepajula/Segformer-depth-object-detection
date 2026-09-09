"""Load HF SegFormer (safetensors) into the MLX model.

Key remap: HF ``transformers`` SegformerForSemanticSegmentation names -> this
package's module tree. Conv weights transpose NCHW-kernel -> NHWC-kernel
((out, in, kH, kW) -> (out, kH, kW, in)).
"""

from __future__ import annotations

import re

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from segformer.model.config import CONFIGS, MiTConfig
from segformer.model.segformer import SegFormer


def _hf_download(repo: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, "model.safetensors")


def _remap_key(k: str) -> str | None:
    """HF key -> MLX module path. Returns None to drop the tensor."""
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

    k = k.replace("decode_head.linear_c.0.proj.", "decode_head.linear_c.0.")
    k = k.replace("decode_head.linear_c.1.proj.", "decode_head.linear_c.1.")
    k = k.replace("decode_head.linear_c.2.proj.", "decode_head.linear_c.2.")
    k = k.replace("decode_head.linear_c.3.proj.", "decode_head.linear_c.3.")
    return k


def _maybe_transpose(mlx_key: str, arr: np.ndarray) -> np.ndarray:
    # 4-D => conv kernel. HF (out, in, kH, kW) -> MLX (out, kH, kW, in).
    if arr.ndim == 4:
        return np.ascontiguousarray(arr.transpose(0, 2, 3, 1))
    return arr


def load_segformer(size: str = "b0", weights_path: str | None = None) -> SegFormer:
    cfg: MiTConfig = CONFIGS[size]
    model = SegFormer(cfg)

    path = weights_path or _hf_download(cfg.hf_repo)
    params: dict[str, mx.array] = {}
    dropped = []
    with safe_open(path, framework="numpy") as f:
        for hf_key in f.keys():
            mlx_key = _remap_key(hf_key)
            if mlx_key is None:
                continue
            arr = _maybe_transpose(mlx_key, f.get_tensor(hf_key))
            params[mlx_key] = mx.array(arr)

    model.load_weights(list(params.items()), strict=True)
    model.eval()  # BatchNorm / any dropout must use inference behaviour
    mx.eval(model.parameters())
    return model

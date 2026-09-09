"""MiT (Mix Transformer) encoder — the SegFormer backbone, in MLX.

Tensor layout: NHWC throughout (MLX conv convention). Token ops run on
(B, N, C); spatial ops (patch embed, attention spatial-reduction, MixFFN
depthwise conv) briefly reshape to (B, H, W, C).

Structure mirrors HF ``transformers`` SegformerModel so its pretrained weights
map key-for-key (see ``weights.py``).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from segformer.model.config import MiTConfig


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch: int, stride: int, in_ch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=stride, padding=patch // 2)
        self.layer_norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array):  # x: (B, H, W, C_in)
        x = self.proj(x)               # (B, H', W', dim)
        _, h, w, c = x.shape
        x = x.reshape(x.shape[0], h * w, c)
        x = self.layer_norm(x)
        return x, h, w


class EfficientAttention(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)  # HF: attention.output.dense
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def _shape(self, x: mx.array, b: int, n: int):
        return x.reshape(b, n, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, h: int, w: int):  # x: (B, N, C)
        b, n, c = x.shape
        q = self._shape(self.query(x), b, n)

        if self.sr_ratio > 1:
            xs = x.reshape(b, h, w, c)
            xs = self.sr(xs)                      # (B, h/r, w/r, C)
            xs = xs.reshape(b, -1, c)
            xs = self.sr_norm(xs)
        else:
            xs = x
        nk = xs.shape[1]
        k = self._shape(self.key(xs), b, nk)
        v = self._shape(self.value(xs), b, nk)

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, c)
        return self.proj(out)


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dense1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.dense2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array, h: int, w: int):  # (B, N, C)
        b, n, _ = x.shape
        x = self.dense1(x)
        hid = x.shape[-1]
        x = x.reshape(b, h, w, hid)
        x = self.dwconv(x)
        x = x.reshape(b, n, hid)
        x = nn.gelu(x)
        return self.dense2(x)


class MiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int, mlp_ratio: int):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, heads, sr_ratio)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, dim * mlp_ratio)

    def __call__(self, x: mx.array, h: int, w: int):
        x = x + self.attn(self.layer_norm_1(x), h, w)
        x = x + self.mlp(self.layer_norm_2(x), h, w)
        return x


class MiTEncoder(nn.Module):
    """Returns 4 feature maps: (B, H/4, W/4, C0) .. (B, H/32, W/32, C3)."""

    def __init__(self, cfg: MiTConfig):
        super().__init__()
        self.cfg = cfg
        dims = cfg.hidden_sizes
        self.patch_embeddings = []
        self.stages = []
        self.stage_norms = []
        in_ch = 3
        for i in range(4):
            self.patch_embeddings.append(
                OverlapPatchEmbed(cfg.patch_sizes[i], cfg.strides[i], in_ch, dims[i])
            )
            blocks = [
                MiTBlock(dims[i], cfg.num_heads[i], cfg.sr_ratios[i], cfg.mlp_ratio)
                for _ in range(cfg.depths[i])
            ]
            self.stages.append(blocks)
            self.stage_norms.append(nn.LayerNorm(dims[i]))
            in_ch = dims[i]

    def __call__(self, x: mx.array):  # x: (B, H, W, 3)
        b = x.shape[0]
        feats = []
        for i in range(4):
            x, h, w = self.patch_embeddings[i](x)
            for blk in self.stages[i]:
                x = blk(x, h, w)
            x = self.stage_norms[i](x)
            x = x.reshape(b, h, w, x.shape[-1])
            feats.append(x)
        return feats

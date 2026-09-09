"""SegFormer all-MLP decode head, in MLX (NHWC).

Each encoder feature -> Linear to a common dim -> upsample to stage-0 (H/4)
resolution -> concat -> 1x1 conv fuse -> BN -> ReLU -> 1x1 conv classifier.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from segformer.model.config import MiTConfig


def _resize_to(x: mx.array, h: int, w: int) -> mx.array:
    """Bilinear resize (B, H0, W0, C) -> (B, h, w, C). align_corners=False."""
    b, h0, w0, c = x.shape
    if (h0, w0) == (h, w):
        return x
    ys = (mx.arange(h, dtype=mx.float32) + 0.5) * (h0 / h) - 0.5
    xs = (mx.arange(w, dtype=mx.float32) + 0.5) * (w0 / w) - 0.5
    ys = mx.clip(ys, 0, h0 - 1)
    xs = mx.clip(xs, 0, w0 - 1)
    y0 = mx.floor(ys).astype(mx.int32)
    x0 = mx.floor(xs).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h0 - 1)
    x1 = mx.minimum(x0 + 1, w0 - 1)
    wy = (ys - y0.astype(mx.float32)).reshape(h, 1, 1)
    wx = (xs - x0.astype(mx.float32)).reshape(1, w, 1)

    def gather(yi, xi):
        return x[:, yi][:, :, xi]  # (B, h, w, C)

    top = gather(y0, x0) * (1 - wx) + gather(y0, x1) * wx
    bot = gather(y1, x0) * (1 - wx) + gather(y1, x1) * wx
    return top * (1 - wy) + bot * wy


class DecodeHead(nn.Module):
    def __init__(self, cfg: MiTConfig):
        super().__init__()
        d = cfg.decoder_hidden_size
        self.linear_c = [nn.Linear(c, d) for c in cfg.hidden_sizes]
        self.linear_fuse = nn.Conv2d(d * 4, d, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm(d)
        self.classifier = nn.Conv2d(d, cfg.num_classes, kernel_size=1)

    def __call__(self, feats: list[mx.array]) -> mx.array:
        b, h0, w0, _ = feats[0].shape
        proj = []
        for x, lin in zip(feats, self.linear_c):
            bb, h, w, c = x.shape
            y = lin(x.reshape(bb, h * w, c)).reshape(bb, h, w, -1)
            proj.append(_resize_to(y, h0, w0))
        x = mx.concatenate(list(reversed(proj)), axis=-1)  # HF concats stage3..stage0
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = nn.relu(x)
        return self.classifier(x)  # (B, H/4, W/4, num_classes)

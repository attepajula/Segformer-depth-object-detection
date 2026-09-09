"""Assembled SegFormer for semantic segmentation (MLX).

Preprocessing mirrors HF ``SegformerImageProcessor`` defaults: resize the RGB
image to 512x512, scale to [0,1], normalize with ImageNet stats. Logits are
bilinearly upsampled back to the input resolution before argmax.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from segformer.model.config import IMAGENET_MEAN, IMAGENET_STD, MiTConfig
from segformer.model.head import DecodeHead, _resize_to
from segformer.model.mit import MiTEncoder

INPUT_SIZE = 512


class SegFormer(nn.Module):
    def __init__(self, cfg: MiTConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MiTEncoder(cfg)
        self.decode_head = DecodeHead(cfg)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        feats = self.encoder(pixel_values)
        return self.decode_head(feats)  # (B, H/4, W/4, num_classes)

    # -- inference helpers ------------------------------------------------
    def _preprocess(self, rgb_uint8: np.ndarray) -> mx.array:
        # Match HF SegformerImageProcessor: PIL BILINEAR resize to 512x512,
        # scale to [0,1], ImageNet normalize.
        from PIL import Image

        img = Image.fromarray(rgb_uint8).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        x = np.asarray(img, np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        return mx.array(x[None])  # (1, 512, 512, 3)

    def logits(self, rgb_uint8: np.ndarray) -> mx.array:
        """Full-resolution class logits (H, W, num_classes) for one RGB frame."""
        h, w = rgb_uint8.shape[:2]
        out = self(self._preprocess(rgb_uint8))          # (1, 128, 128, K)
        out = _resize_to(out, h, w)                       # (1, H, W, K)
        mx.eval(out)
        return out[0]

    def segment(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """(H, W) int32 ADE20K label map."""
        return np.array(mx.argmax(self.logits(rgb_uint8), axis=-1).astype(mx.int32))

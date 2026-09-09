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

    @property
    def param_dtype(self) -> mx.Dtype:
        return self.encoder.patch_embeddings[0].proj.weight.dtype

    def __call__(self, pixel_values: mx.array) -> mx.array:
        pixel_values = pixel_values.astype(self.param_dtype)
        feats = self.encoder(pixel_values)
        return self.decode_head(feats).astype(mx.float32)  # logits back to fp32

    # -- inference helpers ------------------------------------------------
    def _preprocess(self, rgb_uint8: np.ndarray, *, pil: bool = False) -> mx.array:
        # Resize to 512x512, scale to [0,1], ImageNet-normalize. cv2 by default
        # (fast); PIL matches HF SegformerImageProcessor exactly (parity tests).
        if pil:
            from PIL import Image

            img = np.asarray(
                Image.fromarray(rgb_uint8).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            )
        else:
            import cv2

            img = cv2.resize(rgb_uint8, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        return mx.array(x[None])  # (1, 512, 512, 3)

    def logits(self, rgb_uint8: np.ndarray) -> mx.array:
        """Full-resolution class logits (H, W, num_classes). Slow — materializes
        H*W*150 floats. For inspection / parity tests, not the hot path."""
        h, w = rgb_uint8.shape[:2]
        out = _resize_to(self(self._preprocess(rgb_uint8, pil=True)), h, w)
        mx.eval(out)
        return out[0]

    def segment(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """(H, W) int32 ADE20K label map. argmax at model resolution, then
        nearest-upsample the labels (150x less data than upsampling logits)."""
        import cv2

        h, w = rgb_uint8.shape[:2]
        low = self(self._preprocess(rgb_uint8))                        # (1, hs, ws, 150)
        lab = mx.argmax(low[0], axis=-1).astype(mx.uint16)
        mx.eval(lab)
        return cv2.resize(np.array(lab), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    def class_probs(self, rgb_uint8: np.ndarray, class_ids: list[int]) -> np.ndarray:
        """Softmax probability maps for ``class_ids`` only, at frame resolution.

        (H, W, len(class_ids)) float32. Computed at model resolution then
        bilinearly upsampled. The segmenter thresholds these directly — argmax
        over-includes neighbouring wall for the 'door' class.
        """
        import cv2

        h, w = rgb_uint8.shape[:2]
        low = self(self._preprocess(rgb_uint8))[0]                     # (hs, ws, 150)
        prob_low = mx.softmax(low, axis=-1)[:, :, mx.array(class_ids)]  # (hs, ws, T)
        mx.eval(prob_low)
        prob = cv2.resize(np.array(prob_low), (w, h), interpolation=cv2.INTER_LINEAR)
        return prob[:, :, None] if prob.ndim == 2 else prob

    # kept for tests / inspection
    def predict(self, rgb_uint8: np.ndarray, class_ids: list[int]):
        prob = self.class_probs(rgb_uint8, class_ids)
        # argmax label restricted to background + the requested classes
        import cv2

        h, w = rgb_uint8.shape[:2]
        low = self(self._preprocess(rgb_uint8))[0]
        lab = cv2.resize(np.array(mx.argmax(low, axis=-1).astype(mx.uint16)),
                         (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return lab, prob

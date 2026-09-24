"""SegFormer (MiT) configs + the ADE20K classes we care about.

SegFormer = "SegFormer: Simple and Efficient Design for Semantic Segmentation
with Transformers" (Xie et al., NeurIPS 2021). NOT SeqFormer (video instance
segmentation) — different model, one letter apart.

Semantic segmentation, pretrained on ADE20K (150 classes). We keep the door /
railing / bannister / screen-door / stairs / stairway channels and fold everything else into
background. No training — the pretrained weights already know these classes.
"""

from __future__ import annotations

from dataclasses import dataclass

# re-exported for existing imports; the source of truth is segformer.labels
# (framework-free).
from segformer.labels import (
    ADE20K_NUM_CLASSES,
    ADE20K_TARGETS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

__all__ = [
    "ADE20K_NUM_CLASSES", "ADE20K_TARGETS", "IMAGENET_MEAN", "IMAGENET_STD",
    "MiTConfig", "CONFIGS", "MIT_B0", "MIT_B1", "MIT_B2",
]


@dataclass(frozen=True)
class MiTConfig:
    name: str
    hidden_sizes: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    num_heads: tuple[int, int, int, int]
    sr_ratios: tuple[int, int, int, int] = (8, 4, 2, 1)
    patch_sizes: tuple[int, int, int, int] = (7, 3, 3, 3)
    strides: tuple[int, int, int, int] = (4, 2, 2, 2)
    mlp_ratio: int = 4
    decoder_hidden_size: int = 256
    num_classes: int = ADE20K_NUM_CLASSES


MIT_B0 = MiTConfig(
    name="b0",
    hidden_sizes=(32, 64, 160, 256),
    depths=(2, 2, 2, 2),
    num_heads=(1, 2, 5, 8),
    decoder_hidden_size=256,
)

MIT_B1 = MiTConfig(
    name="b1",
    hidden_sizes=(64, 128, 320, 512),
    depths=(2, 2, 2, 2),
    num_heads=(1, 2, 5, 8),
    decoder_hidden_size=256,
)

MIT_B2 = MiTConfig(
    name="b2",
    hidden_sizes=(64, 128, 320, 512),
    depths=(3, 4, 6, 3),
    num_heads=(1, 2, 5, 8),
    decoder_hidden_size=768,
)

CONFIGS = {"b0": MIT_B0, "b1": MIT_B1, "b2": MIT_B2}

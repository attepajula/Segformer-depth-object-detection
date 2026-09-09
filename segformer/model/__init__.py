"""SegFormer semantic segmentation model (MLX).

  load_segformer("b0")  -> SegFormer   (downloads HF weights, maps to MLX)
  model.segment(rgb)    -> (H, W) int32 ADE20K label map
  model.logits(rgb)     -> (H, W, 150) float logits
"""

from segformer.model.config import ADE20K_TARGETS, CONFIGS, MiTConfig
from segformer.model.segformer import SegFormer
from segformer.model.weights import load_segformer

__all__ = ["ADE20K_TARGETS", "CONFIGS", "MiTConfig", "SegFormer", "load_segformer"]

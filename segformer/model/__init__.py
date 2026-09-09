"""SegFormer semantic segmentation model (MLX).

  load_model()          -> SegFormer   (resolves via the registry, models.toml)
  model.segment(rgb)    -> (H, W) int32 ADE20K label map
  model.logits(rgb)     -> (H, W, 150) float logits

Model files live under ``weights/<name>/`` (or ``$SEGFORMER_MODELS_DIR``);
manage them with ``segformer models …``.
"""

from segformer.model.config import ADE20K_TARGETS, CONFIGS, MiTConfig
from segformer.model.registry import ModelSpec, manifest, resolve
from segformer.model.segformer import SegFormer
from segformer.model.weights import load_model

__all__ = [
    "ADE20K_TARGETS", "CONFIGS", "MiTConfig", "ModelSpec",
    "SegFormer", "load_model", "manifest", "resolve",
]

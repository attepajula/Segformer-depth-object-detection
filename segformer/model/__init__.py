"""SegFormer semantic segmentation model.

  load_model()          -> a model    (resolves via the registry, models.toml)
  model.class_probs(rgb, ids) -> (H, W, len(ids)) float32 probability maps

Lazy: importing this package (or ``segformer.model.config`` /
``segformer.model.registry``) does NOT import a model runtime. ``load_model``,
``SegFormer``, ``CONFIGS`` pull in mlx only when first accessed.
"""

from __future__ import annotations

_LAZY = {
    "load_model": ("segformer.model.weights", "load_model"),
    "SegFormer": ("segformer.model.segformer", "SegFormer"),
    "CONFIGS": ("segformer.model.config", "CONFIGS"),
    "MiTConfig": ("segformer.model.config", "MiTConfig"),
    "ADE20K_TARGETS": ("segformer.labels", "ADE20K_TARGETS"),
    "ModelSpec": ("segformer.model.registry", "ModelSpec"),
    "manifest": ("segformer.model.registry", "manifest"),
    "resolve": ("segformer.model.registry", "resolve"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    try:
        mod_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(mod_name), attr)

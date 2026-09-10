"""SegFormer on PyTorch (Apple MPS, Cuda or CPU).

For a machine without MLX. Uses the HF ``transformers`` SegFormer directly —
same pinned repo as the registry (``models.toml``), loaded through
``transformers`` (which handles both ``.safetensors`` and ``pytorch_model.bin``).

Preprocessing and everything after the network match the MLX backend:
resize to the model's input size, ImageNet-normalise, softmax the target
channels, upsample, then the shared pipeline in
``segformer.backends.postprocess``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from segformer.backends.postprocess import probs_to_result
from stream.frame import Frame
from segformer.labels import ADE20K_TARGETS, IMAGENET_MEAN, IMAGENET_STD
from segformer.segmenter import SegResult, Segmenter

_INPUT = {"b0": 512, "b2": 512, "b1": 512, "b3": 512, "b4": 512, "b5": 640}


class TorchSegmenter(Segmenter):
    name = "torch"

    def __init__(self, model: str | None = None, *, min_area: int = 600,
                 use_depth: bool = True, smooth: bool = True,
                 targets: dict[int, str] | None = None):
        self.model_name = model
        self.min_area = min_area
        self.use_depth = use_depth
        self.targets = targets or ADE20K_TARGETS
        self._net = None
        self._device = None
        self._input = 512
        self._tracker = None
        if smooth:
            from segformer.tracking import Tracker

            self._tracker = Tracker()

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()

    def load(self, checkpoint: str | None = None) -> None:
        if self._net is not None:
            return
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation
        except ImportError as e:
            raise ImportError(
                "torch backend needs torch + transformers:\n"
                "  uv pip install --python .venv/bin/python -e '.[torch]'"
            ) from e

        from segformer.model.registry import get

        spec = get(self.model_name)
        self._input = _INPUT.get(spec.arch, 512)
        src = checkpoint or spec.repo
        kw = {} if checkpoint else {"revision": spec.revision}
        self._net = SegformerForSemanticSegmentation.from_pretrained(src, **kw).eval()
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self._net.to(self._device)

    def _class_probs(self, rgb: np.ndarray, ids: list[int]) -> np.ndarray:
        import cv2
        import torch
        import torch.nn.functional as F

        h, w = rgb.shape[:2]
        img = cv2.resize(rgb, (self._input, self._input), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._net(pixel_values=t).logits          # (1, 150, H/4, W/4)
            p = logits.softmax(dim=1)[:, ids]                    # (1, T, H/4, W/4)
            p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
        return p[0].permute(1, 2, 0).float().cpu().numpy()       # (H, W, T)

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        ids = list(self.targets)
        prob = self._class_probs(frame.rgb, ids)
        res = probs_to_result(frame, prob, ids, min_area=self.min_area,
                              use_depth=self.use_depth, tracker=self._tracker)
        return res, state

    def close(self) -> None:
        self._net = None

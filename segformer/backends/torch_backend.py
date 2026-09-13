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
        self._dtype = None
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

        from segformer.model.registry import ensure_torch_source, get

        spec = get(self.model_name)
        self._input = _INPUT.get(spec.arch, 512)
        if checkpoint:
            self._net = SegformerForSemanticSegmentation.from_pretrained(checkpoint).eval()
        elif spec.file.endswith(".safetensors"):
            # already safetensors on the pinned revision -- transformers reads
            # it directly, no .bin/pickle path involved.
            self._net = SegformerForSemanticSegmentation.from_pretrained(
                spec.repo, revision=spec.revision).eval()
        else:
            # repo only ships pytorch_model.bin: ensure_torch_source converts
            # it to safetensors once (sha256-verified source, never re-read
            # after) and caches the pinned config alongside it, then
            # from_pretrained loads that local directory -- still going
            # through transformers' own old-checkpoint key renaming, unlike
            # a raw load_state_dict.
            local_dir = ensure_torch_source(spec)
            self._net = SegformerForSemanticSegmentation.from_pretrained(local_dir).eval()
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        # fp16 halves compute + bandwidth on GPU; CPU has no fast fp16 kernels
        # for these ops, so stay fp32 there.
        self._dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._net.to(self._device, dtype=self._dtype)

    def _class_probs(self, rgb: np.ndarray, ids: list[int]) -> np.ndarray:
        import cv2
        import torch
        import torch.nn.functional as F

        h, w = rgb.shape[:2]
        img = cv2.resize(rgb, (self._input, self._input), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        t = torch.from_numpy(x).permute(2, 0, 1).contiguous().unsqueeze(0)
        t = t.to(self._device, dtype=self._dtype)

        with torch.no_grad():
            logits = self._net(pixel_values=t).logits          # (1, 150, H/4, W/4)
            p = logits.float().softmax(dim=1)[:, ids]            # softmax in fp32: stable, cheap (T << 150)
            p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
        return p[0].permute(1, 2, 0).contiguous().cpu().numpy()  # (H, W, T), already fp32

    def infer(self, frame: Frame, state: Any | None) -> tuple[SegResult, Any]:
        ids = list(self.targets)
        prob = self._class_probs(frame.rgb, ids)
        res = probs_to_result(frame, prob, ids, min_area=self.min_area,
                              use_depth=self.use_depth, tracker=self._tracker)
        return res, state

    def close(self) -> None:
        self._net = None

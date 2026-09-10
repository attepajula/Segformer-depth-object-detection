"""segformer — door / railing semantic segmentation.

Consumes an RGB-D frame stream (the ``stream`` package: contract, codec,
recording, replay, subscribe) and runs SegFormer (MiT encoder + all-MLP head,
ADE20K-pretrained) behind a framework-agnostic
:class:`~segformer.segmenter.Segmenter` interface — MLX or PyTorch backend.

Not SeqFormer (video instance segmentation) — different model, one letter off.
"""

__version__ = "0.0.1"

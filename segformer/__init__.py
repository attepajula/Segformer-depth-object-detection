"""segformer — door / railing semantic segmentation over the xtion RGB-D stream.

Consumes the device-agnostic frame contract (schema v1) from the ``xtion``
capture pipeline over ZeroMQ. Model: SegFormer (MiT encoder + all-MLP head),
ADE20K-pretrained, ported to MLX. Behind a framework-agnostic
:class:`~segformer.segmenter.Segmenter` interface.

Not SeqFormer (video instance segmentation) — different model, one letter off.
"""

__version__ = "0.0.1"

WIRE_SCHEMA_VERSION = 1  # frame contract version this consumer understands

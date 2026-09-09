"""seqformer — video instance segmentation over the xtion RGB-D stream.

Consumes the device-agnostic frame contract (schema v1) published by the
``xtion`` capture pipeline over ZeroMQ. The segmentation model sits behind a
framework-agnostic :class:`~seqformer.segmenter.Segmenter` interface so a
PyTorch/MPS and an MLX implementation are interchangeable.

Status: scaffold only. Wire decode + interfaces are defined; the model,
backends, and CLI are stubs.
"""

__version__ = "0.0.1"

WIRE_SCHEMA_VERSION = 1  # frame contract version this consumer understands

"""stream — the device-agnostic RGB-D frame stream.

Everything about *getting frames*, independent of the segmentation net: the
frame contract, the on-wire codec, the recording format, replay, and the
subscribe side. Imports nothing from ``segformer``.

  from stream import Frame, Intrinsics, decode_frame, encode_frame
  from stream import FrameConsumer            # SUB side
  from stream import RecordingReader, replay  # play a recorded session

Schema is versioned: a frame or recording whose ``schema_version`` differs from
``WIRE_SCHEMA_VERSION`` is rejected, not misread. This mirrors the capture
service's contract (`../xtion`); the two are deliberately separate copies of a
frozen format.
"""

from stream.frame import WIRE_SCHEMA_VERSION, Frame, Intrinsics
from stream.wire import WireError, decode_frame, decode_header, encode_frame
from stream.consumer import DEFAULT_ENDPOINT, FrameConsumer, RateMeter
from stream.recording import RecordingReader
from stream import replay

__all__ = [
    "WIRE_SCHEMA_VERSION",
    "Frame", "Intrinsics",
    "WireError", "decode_frame", "decode_header", "encode_frame",
    "DEFAULT_ENDPOINT", "FrameConsumer", "RateMeter",
    "RecordingReader", "replay",
]

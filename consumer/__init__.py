"""Downstream consumers of segformer detections.

Separate from the net: takes ``(Frame, SegResult)`` pairs and does something
with them. Imports only ``stream`` and the framework-free ``segformer.labels``.
"""

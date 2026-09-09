#!/usr/bin/env python3
"""Numerically validate the MLX SegFormer port against HF transformers (CPU).

Not part of the test suite (needs torch + transformers). Run once after any
change to the model or weight mapping.

    .venv/bin/python tools/compare_hf.py [--size b0] [--image path.png]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="b0")
    ap.add_argument("--own-preproc", action="store_true", help="feed identical tensor to HF")
    ap.add_argument("--image", default=None, help="RGB image; default = a session1 frame")
    args = ap.parse_args()

    if args.image:
        import cv2

        rgb = cv2.imread(args.image)[:, :, ::-1].copy()
    else:
        rgb = np.load("/Users/porpyla/projut/xtion/session1/chunk_00003.npz")["rgb"][5]

    # --- HF reference -------------------------------------------------
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    repo = {"b0": "nvidia/segformer-b0-finetuned-ade-512-512",
            "b2": "nvidia/segformer-b2-finetuned-ade-512-512"}[args.size]
    hf = SegformerForSemanticSegmentation.from_pretrained(repo).eval()

    # --- MLX port ---------------------------------------------------
    import mlx.core as mx

    from segformer.model import load_segformer

    model = load_segformer(args.size)
    pre = model._preprocess(rgb)                    # (1,512,512,3) NHWC, our pipeline
    mlx_low = model(pre)

    if args.own_preproc:
        # feed the SAME tensor to HF -> isolates model-port error from preprocessing
        px = torch.from_numpy(np.array(pre)).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            hf_logits = hf(pixel_values=px).logits
    else:
        proc = SegformerImageProcessor.from_pretrained(repo)
        with torch.no_grad():
            hf_logits = hf(**proc(images=rgb, return_tensors="pt")).logits
    hf_logits = hf_logits[0].permute(1, 2, 0).numpy()

    mx.eval(mlx_low)
    mlx_logits = np.array(mlx_low)[0]

    # --- compare (low-res logits, pre-upsample) --------------------
    d = np.abs(hf_logits - mlx_logits)
    hf_lab = hf_logits.argmax(-1)
    mlx_lab = mlx_logits.argmax(-1)
    agree = (hf_lab == mlx_lab).mean()
    print(f"size={args.size}  logits |Δ|  mean={d.mean():.4e}  max={d.max():.4e}")
    print(f"argmax agreement (128x128): {agree*100:.2f}%")
    print(f"cos sim: {np.sum(hf_logits*mlx_logits)/(np.linalg.norm(hf_logits)*np.linalg.norm(mlx_logits)):.6f}")

    # --own-preproc isolates the model port (should be bit-exact);
    # default path also folds in PIL-resize resampling differences.
    if args.own_preproc:
        ok = d.mean() < 1e-3 and agree > 0.999
    else:
        ok = d.mean() < 0.05 and agree > 0.99
    print("PASS" if ok else "FAIL — port diverges")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

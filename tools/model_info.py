#!/usr/bin/env python3
"""Print everything needed to register one HF SegFormer repo — both manual
steps, not just the manifest stanza.

    .venv/bin/python tools/model_info.py nvidia/segformer-b3-finetuned-ade-512-512

`HfApi().model_info()` returns a huge object; this pulls out exactly what
`models.toml` needs and prints a ready-to-paste `[models.<name>]` block.

Separately: the **mlx backend** also needs the architecture's dimensions
(hidden_sizes/depths/num_heads/decoder_hidden_size) in
`segformer/model/config.py`'s `CONFIGS` dict — a `models.toml` stanza alone
isn't enough for mlx if `arch` isn't already there (only b0/b1/b2 are, as of
this writing; torch backend needs neither `CONFIGS` nor this second step, it
reads the HF repo's own `config.json` directly). If missing, this also fetches
`config.json` and prints a ready-to-paste `MiTConfig`.
"""
from __future__ import annotations

import argparse
import sys

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="HF repo id, e.g. nvidia/segformer-b1-finetuned-ade-512-512")
    ap.add_argument("--name", default=None, help="stanza name (default: derived from repo)")
    ap.add_argument("--arch", default=None, help="b0/b1/b2/... (default: guessed from repo id)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(args.repo, files_metadata=True)
    by_name = {s.rfilename: s for s in info.siblings}
    weight = next((by_name[f] for f in _WEIGHT_FILES if f in by_name), None)
    if weight is None:
        print(f"no {_WEIGHT_FILES} found in {args.repo}; files: "
              f"{[s.rfilename for s in info.siblings]}", file=sys.stderr)
        return 1
    if weight.lfs is None:
        print(f"{weight.rfilename} isn't an LFS file (no sha256) — can't pin it", file=sys.stderr)
        return 1

    name = args.name or f"segformer-{args.repo.split('/')[-1].split('-')[1]}-ade"
    arch = args.arch or next((p for p in args.repo.split("-") if p.startswith("b") and p[1:].isdigit()), "b0")

    print(f"[models.{name}]")
    print(f'description = "SegFormer-{arch.upper()}, ADE20K-150 semantic segmentation (NVIDIA)"')
    print(f'arch        = "{arch}"')
    print("input       = 512")
    print('classes     = "ade20k-150"')
    print('source      = "hf"')
    print(f'repo        = "{args.repo}"')
    print(f'revision    = "{info.sha}"')
    print(f'file        = "{weight.rfilename}"')
    print(f'sha256      = "{weight.lfs.sha256}"')
    print(f"size        = {weight.size}")

    try:
        from segformer.model.config import CONFIGS
    except ImportError:
        CONFIGS = {}  # mlx not installed here -- can't check, fall through to print it anyway

    if arch in CONFIGS:
        print(f"\n# arch {arch!r} already in segformer/model/config.py -- nothing else to do for mlx.")
        return 0

    print(f"\n# --- mlx backend also needs this in segformer/model/config.py ---")
    print(f"# arch {arch!r} is NOT in CONFIGS yet. Paste this in, add \"{arch}\": MIT_{arch.upper()}")
    print(f"# to the CONFIGS dict. (torch backend needs none of this.)")
    try:
        cfg_path = hf_hub_download(args.repo, "config.json", revision=info.sha)
        import json

        c = json.load(open(cfg_path))
        hs = tuple(c["hidden_sizes"])
        d = tuple(c["depths"])
        nh = tuple(c["num_attention_heads"])
        dec = c["decoder_hidden_size"]
        print(f"MIT_{arch.upper()} = MiTConfig(")
        print(f'    name="{arch}",')
        print(f"    hidden_sizes={hs},")
        print(f"    depths={d},")
        print(f"    num_heads={nh},")
        print(f"    decoder_hidden_size={dec},")
        print(")")
    except Exception as e:  # noqa: BLE001
        print(f"# couldn't fetch config.json to fill this in automatically: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

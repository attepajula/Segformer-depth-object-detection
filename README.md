# segformer

Semantic segmentation of **doors and railings** in the RGB-D stream from the
[`xtion`](../xtion) capture pipeline. **MLX** (Apple GPU, native arm64).

Model: **SegFormer** (Xie et al., NeurIPS 2021) — MiT transformer encoder +
all-MLP decode head. Pretrained on **ADE20K**, which already has `door`,
`railing`, `bannister`, `screen door` as classes — no training. Ported to MLX
and validated bit-exact against HF `transformers` (`tools/compare_hf.py`).

> Not *SeqFormer* (video instance segmentation). One letter apart. Doors and
> railings are static structure — semantic segmentation, not instance tracking.

## Pipeline

```
xtion capture ──ZMQ PUB (schema v1)──▶  segformer run
                tcp://127.0.0.1:5556      wire.decode ─▶ MlxSegmenter ─▶ overlay
                                          SegFormer-B0, ADE20K → door/railing masks
                                          → depth gate (geometry.py)
```

- **No coupling to the capture repo** — `segformer/wire.py` + `frame.py` are an
  independent copy of the schema-v1 contract, guarded by `WIRE_SCHEMA_VERSION`.
- **Framework-agnostic** — everything talks to `segformer.segmenter.Segmenter` /
  `SegResult`. Backends: `mlx` (primary), `torch-mps` (secondary, stub),
  `dummy` (no model).
- **Depth gate** (`geometry.py`): B0 (37 mIoU) calls large flat surfaces
  "door". The gate back-projects the mask to metric 3D and rejects anything
  that fills the frame, isn't door-sized (~0.5–1.5 m wide, >1.5 m tall), isn't
  planar, or isn't vertical. `MlxSegmenter(use_depth=True)`, on by default.

## Setup (Apple Silicon, native arm64)

```sh
uv venv --python python3.14 .venv
uv pip install --python .venv/bin/python -e ".[mlx,viewer,dev]"
.venv/bin/python -m pytest -q

# optional: HF-parity check (installs torch + transformers, CPU)
uv pip install --python .venv/bin/python -e ".[reference]"
.venv/bin/python tools/compare_hf.py                  # full pipeline
.venv/bin/python tools/compare_hf.py --own-preproc    # model port only (bit-exact)
```

## Model management

Registry, manifest in `segformer/models.toml`. No `huggingface_hub` — the
registry does one pinned HTTPS GET + sha256 check, then a one-time convert to an
MLX-ready file. Runtime imports neither `transformers` nor `safetensors`.

```sh
PY=.venv/bin/python
$PY -m segformer models list          # names, arch, classes, state (-|source|ready)
$PY -m segformer models pull          # download the pinned source safetensors
$PY -m segformer models convert       # + build weights/<name>/model.mlx.safetensors
$PY -m segformer models verify        # re-hash source, load artifact
$PY -m segformer models path          # print the artifact path
$PY -m segformer models where         # print the cache dir
```

Files land in `weights/<name>/` (`source.safetensors`, `model.mlx.safetensors`,
`model.json` provenance sidecar). Override the dir with `$SEGFORMER_MODELS_DIR`.
`pull` reuses an existing `~/.cache/huggingface` download if present.
`run`/`bench` auto-`convert` on first use. A fine-tune: add a `source = "local"`
entry, drop its file in `weights/<name>/`, done.

## Usage

```sh
PY=.venv/bin/python
$PY -m segformer probe                         # decode one frame, print stats
$PY -m segformer run --backend mlx --overlay    # live door/railing segmentation
$PY -m segformer run --backend mlx --save out/  # headless, PNG per frame
$PY -m segformer bench --backend mlx            # fps
```

Test rig (no camera):
```sh
xtion/.venv/bin/python -m capture replay xtion/session1/ --publish --loop
```
then `segformer run --backend mlx --overlay`.

## Status

| part | state |
|---|---|
| `wire.py`, `frame.py` | done — schema-v1 decode |
| `segmenter.py`, `consumer.py`, `overlay.py` | done |
| `model/` — MiT encoder + decode head (MLX) | done, HF-validated bit-exact |
| `model/registry.py` + `models.toml` | done — pinned, sha256, offline-first |
| `model/convert.py` — HF safetensors → MLX artifact | done |
| `backends/mlx_backend.py` | done — score gate + depth gate + connected components |
| `backends/torch_mps.py` | stub — secondary |
| `geometry.py` — depth door gate | done (needs a real-door scene to tune true positives) |
| `bench` command | done |
| perf: infer 42 ms / 24 fps (fp16, cv2 preprocess, model-res argmax/softmax) | ok |

## Known limits

- **B0 accuracy**: ~37 mIoU. Precision on doors comes from the depth gate, not
  the model. b1–b5 ship only `pytorch_model.bin` (no safetensors) — a bigger
  model needs a torch-based conversion into the registry.
- **Depth gate true positives unverified** — every test scene so far has *no*
  real door, so the gate rejecting everything B0 flags is correct but untested
  on the keep path. Needs footage with an actual door.
- **Live fps ~15–18** (< the 24 fps bench): decode + overlay + `imshow`, and
  `capture serve` sharing the machine under Rosetta. A threaded consumer would
  decouple display from throughput.

## Next

1. Railing depth cue: thin-structure / see-through-gap density, height band.
2. Temporal EMA on masks — cheap stability for a static scene.
3. Threaded consumer (infer thread + display thread) if live fps matters.
4. Bigger model via a torch `.bin` → MLX conversion path in the registry.

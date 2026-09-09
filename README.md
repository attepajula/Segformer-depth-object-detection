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
- **Precision stack** (B0 is only 37 mIoU, so filtering does the work):
  1. threshold the door *probability* map at 0.85, not argmax — argmax eats the
     wall around the door;
  2. depth gate (`geometry.py`) — back-project the mask, require a door-sized
     (~0.4–1.9 m wide), planar, roughly vertical structure; tolerant of a door
     only partly in frame;
  3. temporal tracker (`tracking.py`) — 3 hits to confirm, lingers 6 frames;
     kills the frame-to-frame flicker.

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
| `backends/mlx_backend.py` | done — prob-threshold + depth gate + tracker |
| `backends/torch_mps.py` | stub — secondary |
| `geometry.py` — depth door gate | done, tuned on session2 (one real door) |
| `tracking.py` — temporal hysteresis | done |
| `bench` command | done |
| perf: infer 42 ms / 24 fps (fp16, cv2 preprocess, model-res argmax/softmax) | ok |

## Known limits

- **B0 accuracy**: ~37 mIoU. Precision on doors comes from the depth gate, not
  the model. b1–b5 ship only `pytorch_model.bin` (no safetensors) — a bigger
  model needs a torch-based conversion into the registry.
- **One real door, tuned on it** — session2 has a single door (short USB
  tether, door barely fits the frame). On a 250-frame segment with it in view:
  detected ~96% of frames, 2 on/off flips. session1 (no door): 0 false doors.
  Not validated on multiple doors / door types / open doors.
- **Live fps ~15–18** (< the 24 fps bench): decode + overlay + `imshow`, and
  `capture serve` sharing the machine under Rosetta. A threaded consumer would
  decouple display from throughput.

## Next

1. Railing: B0 barely sees railings (2 blobs in all of session2). Needs a
   bigger model or a depth-primary detector (thin repeated vertical structure).
2. Threaded consumer (infer thread + display thread) if live fps matters.
3. Bigger model via a torch `.bin` → MLX conversion path in the registry.
4. Split door-from-wall when B0 does merge them (depth plane discontinuity).

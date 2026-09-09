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
                                          (depth post-process: TODO)
```

- **No coupling to the capture repo** — `segformer/wire.py` + `frame.py` are an
  independent copy of the schema-v1 contract, guarded by `WIRE_SCHEMA_VERSION`.
- **Framework-agnostic** — everything talks to `segformer.segmenter.Segmenter` /
  `SegResult`. Backends: `mlx` (primary), `torch-mps` (secondary, stub),
  `dummy` (no model).
- **Depth**: planned as a post-process — door = wall-flush plane, railing =
  thin structure at a height. Model stays RGB-only. Not built yet.

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

Weights download automatically from `nvidia/segformer-b0-finetuned-ade-512-512`
on first use (~15 MB), cached by `huggingface_hub`.

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
| `model/weights.py` — HF safetensors → MLX | done (B0, B2) |
| `backends/mlx_backend.py` | done — ADE targets → connected-component instances |
| `backends/torch_mps.py` | stub — secondary |
| `bench` command | done |
| depth post-process (`geometry.py`) | **not built** |
| perf: post-processing is CPU numpy, ~150 ms/frame (inference 77 ms) | to optimize |

## Next

1. Depth post-process: validate door masks against plane fit, railing against
   thin-structure + height; drop false positives, close mask holes.
2. Perf: argmax on GPU, per-target-class prob only, skip full 150-way softmax.
3. Temporal smoothing (EMA on masks) — cheap stability win for a static scene.
4. Try B2 if B0 accuracy is short.

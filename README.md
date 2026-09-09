# seqformer

Video instance segmentation over the RGB-D stream from the [`xtion`](../xtion)
capture pipeline. **MLX is the primary backend** (Apple GPU, native arm64).

Working: wire decode, `Segmenter` interface, consumer loop, overlay, and
`seqformer run` end-to-end with `DummySegmenter`. Not built: the real SeqFormer
model (MLX and torch-mps segmenters are stubs).

## Design

```
xtion capture ──ZMQ PUB (schema v1)──▶  seqformer run
                tcp://127.0.0.1:5556      wire.decode ─▶ Segmenter ─▶ overlay / tracking
```

- **No coupling to the capture repo.** `seqformer/wire.py` + `seqformer/frame.py`
  are an independent copy of the schema-v1 frame contract. `WIRE_SCHEMA_VERSION`
  guards it — a mismatched frame is rejected, not misread.
- **Framework-agnostic model.** Everything talks to `seqformer.segmenter.Segmenter`
  / `SegResult`. Two implementations, chosen at construction:
  - `torch-mps` — PyTorch, Apple GPU via MPS. MS-deform-attn uses the pure-Python
    fallback (fused kernel is CUDA-only). Loads official SeqFormer checkpoints.
  - `mlx` — MLX, Apple GPU, unified-memory zero-copy from frames. Needs a
    pure-MLX or custom-Metal MS-deform-attn (not written) + checkpoint conversion.
  - `dummy` — no model; emits the valid-depth region as one instance. For wiring.

## Setup (Apple Silicon)

Native arm64, unlike the capture process (which is x86_64/Rosetta for OpenNI2).
Uses the system Homebrew Python (3.14); torch and mlx both ship 3.14 wheels.

```sh
uv venv --python python3.14 .venv
uv pip install --python .venv/bin/python -e ".[mlx,viewer,dev]"
# torch-mps is optional / secondary:
# uv pip install --python .venv/bin/python -e ".[torch]"

.venv/bin/python -m pytest -q          # tests, no model, no hardware
```

## Usage

```sh
PY=.venv/bin/python

$PY -m seqformer probe                              # decode one frame, print stats
$PY -m seqformer run --backend dummy --overlay      # end-to-end, OpenCV window
$PY -m seqformer run --backend dummy --save out/    # headless, write overlay PNGs
$PY -m seqformer run --backend mlx                  # NotImplementedError (model not built)
$PY -m seqformer bench --backend mlx                # stub
```

Test rig (no camera): from the `xtion` repo,
```sh
xtion/.venv/bin/python -m capture replay xtion/session1/ --publish --loop
```
then `seqformer run --backend dummy --overlay`.

## Layout

| path | state |
|---|---|
| `seqformer/wire.py`, `frame.py` | done — schema-v1 decode, independent of capture |
| `seqformer/segmenter.py` | done — `Segmenter` ABC, `SegResult`, `Instance` |
| `seqformer/consumer.py`, `overlay.py` | done — ZMQ SUB/CONFLATE loop + mask/box render |
| `seqformer/backends/dummy.py` | done — no-model segmenter |
| `seqformer/backends/mlx_backend.py` | stub — plan + open problems in docstring |
| `seqformer/backends/torch_mps.py` | stub — secondary backend |
| `seqformer/model/` | notes only — architecture breakdown |
| `seqformer/cli.py` | `probe` + `run` wired; `bench` stub |

## Next (MLX model build order)

1. **ResNet-50 backbone** in `mlx.nn` → C3-C5 features. Convert torchvision
   weights, verify feature maps on GPU. `seqformer/model/backbone.py`.
2. **Input projection + positional encoding** → 256-d tokens.
3. **MS-deformable-attention**: pure-MLX first (gather + bilinear sample at
   offsets + softmax weights). Correctness vs a reference before optimizing.
4. **Deformable encoder/decoder** + instance queries.
5. **SeqFormer heads**: class (clip-level), box (per-frame), mask (dynamic conv).
6. **Checkpoint conversion**: official SeqFormer PyTorch `state_dict` → MLX.
7. Wire into `MlxSegmenter.infer`, `seqformer bench`. Measure fps.
8. If deform-attn is the bottleneck → `mx.fast.metal_kernel` custom kernel.

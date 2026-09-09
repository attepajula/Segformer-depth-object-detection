# seqformer

Video instance segmentation over the RGB-D stream from the [`xtion`](../xtion)
capture pipeline. **Scaffold only** — wire decode and interfaces are done; the
model, backends, and consumer loop are stubs.

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

```sh
uv python install cpython-3.13-macos-aarch64
uv venv --python cpython-3.13-macos-aarch64 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# model frameworks (heavy, install when needed):
uv pip install --python .venv/bin/python -e ".[torch]"   # torch + torchvision
uv pip install --python .venv/bin/python -e ".[mlx]"     # mlx

.venv/bin/python -m pytest -q          # 6 tests, no model, no hardware
```

## Usage

```sh
PY=.venv/bin/python

# works now — subscribe, decode one frame, print stats:
$PY -m seqformer probe --endpoint tcp://127.0.0.1:5556

# stubs (NotImplementedError):
$PY -m seqformer run  --backend torch-mps --overlay
$PY -m seqformer bench --backend mlx --frames 300
```

Test it against a live stream: from the `xtion` repo run
`.venv/bin/python -m capture serve --device fake`, then `seqformer probe`.

## Layout

| path | state |
|---|---|
| `seqformer/wire.py`, `frame.py` | done — schema-v1 decode, independent of capture |
| `seqformer/segmenter.py` | done — `Segmenter` ABC, `SegResult`, `Instance` |
| `seqformer/backends/dummy.py` | done — no-model segmenter for wiring |
| `seqformer/backends/torch_mps.py` | stub — plan in docstring |
| `seqformer/backends/mlx_backend.py` | stub — plan + open problems in docstring |
| `seqformer/model/` | notes only — architecture breakdown |
| `seqformer/cli.py` | `probe` wired; `run` / `bench` stubs |

## Next

1. `seqformer run`: consumer loop (ZMQ SUB CONFLATE) → `DummySegmenter` → OpenCV
   overlay. Proves the full path.
2. `TorchMpsSegmenter`: ResNet-50 backbone + Deformable-DETR + SeqFormer head,
   pure-Python deform-attn, load official weights.
3. Measure fps on MPS. If too slow → `MlxSegmenter` + Metal deform-attn kernel.

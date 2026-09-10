# segformer

Semantic segmentation of **doors and railings** in the RGB-D stream from the
[`xtion`](../xtion) capture pipeline. **MLX** on Apple Silicon, or **PyTorch**
(CUDA / MPS / CPU) on any other machine.

Model: **SegFormer** (Xie et al., NeurIPS 2021) — MiT transformer encoder +
all-MLP decode head. Pretrained on **ADE20K**, which already has `door`,
`railing`, `bannister`, `screen door` as classes — no training. Ported to MLX
and validated bit-exact against HF `transformers` (`tools/compare_hf.py`).



## Pipeline

```
 ┌─ xtion  capture serve  (live camera; macOS + Xtion + OpenNI2/Rosetta)
 ├─ xtion  capture replay <session>          ──┐
 └─ segformer  replay <session>  (this repo) ──┤  same schema-v1 ZMQ stream
                                               ▼
                              segformer run  ─▶ wire.decode ─▶ segmenter ─▶ overlay
                              tcp://127.0.0.1:5556  SegFormer-B0, ADE20K
                              (mlx | torch backend) → door/railing masks → depth gate
```

### `stream/` vs the `xtion` repo — pick one producer

Both implement the **same schema-v1 frame stream**; they don't run together.

| | `xtion` repo (`capture/`) | this repo (`stream/`) |
|---|---|---|
| needs | the Xtion camera, OpenNI2, macOS x86_64 / Rosetta | nothing — `numpy / pyzmq / msgpack` |
| does | record sessions, `capture serve` (live), `capture replay` | `segformer replay` (play a recording) |
| use when | you have the camera on the Mac | you only have recordings (a collaborator) |

The two `replay`s are equivalent — a recording published by either is byte-identical on the wire. `stream/` is a deliberate, standalone copy of the frozen contract (`WIRE_SCHEMA_VERSION`), so a collaborator never clones `xtion`. If the schema ever bumps, both copies change together.

- **`stream/` — the frame stream, separate from the net.** Contract
  (`stream/frame.py`), codec (`stream/wire.py`), recording format
  (`stream/recording.py`), replay (`stream/replay.py`), subscribe
  (`stream/consumer.py`). Imports nothing from `segformer`. See below on how it
  relates to the `xtion` repo.
- **Framework-agnostic** — everything talks to `segformer.segmenter.Segmenter` /
  `SegResult`. Backends: `mlx` (Apple GPU, primary), `torch` (HF transformers on
  CUDA / MPS / CPU — for a machine without MLX), `dummy` (no model). Same
  probability→mask→depth-gate→tracker pipeline after the network
  (`backends/postprocess.py`); the backends differ only in producing the
  probability maps. Pick with `--backend`, `$SEGFORMER_BACKEND`, else `mlx`.
- **Precision stack** (B0 is only 37 mIoU, so filtering does the work):
  1. threshold the door *probability* map at 0.85, not argmax — argmax eats the
     wall around the door;
  2. depth-consolidate — B0 emits ~5 fragmented door blobs; merge the ones on
     the same depth plane into one region (`geometry.consolidate_door`);
  3. depth gate (`geometry.door_gate`) — back-project, require a door-sized
     (~0.4–1.9 m wide), planar, roughly vertical structure; tolerant of a door
     only partly in frame — runs **once** on the consolidated region;
  4. temporal tracker (`tracking.py`) — 3 hits to confirm, lingers 6 frames;
     kills the frame-to-frame flicker.
- **Threaded run loop** (`runner.py`) — inference on a worker thread, `imshow`
  on the main thread. Overlay + display hide behind the next inference instead
  of adding to it. `run --no-thread` for the serial loop.

## Setup

```sh
uv venv --python python3.14 .venv

# Apple Silicon (MLX backend):
uv pip install --python .venv/bin/python -e ".[mlx,viewer,dev]"

# no MLX (torch backend — CUDA / MPS / CPU):
uv pip install --python .venv/bin/python -e ".[torch,viewer,dev]"
#   then either export SEGFORMER_BACKEND=torch  or pass --backend torch
#   NVIDIA: install the CUDA torch wheel first, else it falls back to CPU:
#   uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124

.venv/bin/python -m pytest -q
```

The `[torch]` extra also powers the HF-parity check:
```sh
.venv/bin/python tools/compare_hf.py                  # full pipeline
.venv/bin/python tools/compare_hf.py --own-preproc    # MLX port vs HF, bit-exact
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

### Working from recordings (no camera, no xtion repo)

`segformer replay` publishes a recorded session on ZeroMQ — indistinguishable
from a live camera to `segformer run`. This is all the capture side a
collaborator needs; no OpenNI2, no Rosetta, no `xtion` checkout.

```sh
segformer replay session2/ --info            # frames, duration, size on disk
segformer replay session2/ --loop            # terminal 1: publish (original timing)
segformer run --backend torch --overlay      # terminal 2: consume + segment
```

Recordings are large (GB/minute) and not in git — move them out of band. Format
+ details: [`docs/recording-format.md`](docs/recording-format.md).

## Status

| part | state |
|---|---|
| `stream/` — frame contract, codec, recording, replay, subscribe | done — independent of the net |
| `segmenter.py`, `overlay.py` | done |
| `model/` — MiT encoder + decode head (MLX) | done, HF-validated bit-exact |
| `model/registry.py` + `models.toml` | done — pinned, sha256, offline-first |
| `model/convert.py` — HF safetensors → MLX artifact | done |
| `backends/postprocess.py` | done — shared prob→mask→gate→tracker pipeline |
| `backends/mlx_backend.py` | done — MLX SegFormer → probability maps |
| `backends/torch_backend.py` | done — HF transformers on CUDA / MPS / CPU → probability maps |
| `geometry.py` — depth consolidate + door gate | done, tuned on session2 (one real door) |
| `tracking.py` — temporal hysteresis | done |
| `runner.py` — threaded run loop | done |
| `bench` command | done |
| perf: infer 29 ms no-door / 39 ms door; threaded loop ~26 fps effective | ok |

## Known limits

- **B0 accuracy**: ~37 mIoU. Precision on doors comes from the depth gate, not
  the model. b1–b5 ship only `pytorch_model.bin` (no safetensors) — a bigger
  model needs a torch-based conversion into the registry.
- **One real door, tuned on it** — session2 has a single door (short USB
  tether, door barely fits the frame). On a 250-frame segment with it in view:
  detected ~96% of frames, 2 on/off flips. session1 (no door): 0 false doors.
  Not validated on multiple doors / door types / open doors.
- **fps floor is the model** — ~22 ms forward for B0 at 512² on the GPU,
  memory-bandwidth-bound (fp16 gives nothing). +5–7 ms preprocess + softmax,
  +~10 ms depth path on door frames. The threaded loop hides decode/overlay/
  imshow behind inference, so effective ~26 fps; `capture serve` (Rosetta)
  sharing the machine still costs a few fps live.

## Next

1. Railing: B0 barely sees railings (2 blobs in all of session2). Needs a
   bigger model or a depth-primary detector (thin repeated vertical structure).
2. Bigger model via a torch `.bin` → MLX conversion path in the registry
   (b1–b5 have no safetensors). Or a smaller model input (384²) for ~15 ms fwd.
3. Split door-from-wall when B0 merges them beyond depth-consolidation
   (a depth-plane discontinuity within the merged region).

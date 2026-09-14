# segformer

Semantic segmentation of **doors and railings** in the RGB-D stream from the
[`xtion`](../xtion) capture pipeline. Model: **SegFormer** (Xie et al.,
NeurIPS 2021), pretrained on **ADE20K** (already has `door`, `railing`,
`bannister`, `screen door` — no training needed). Two interchangeable
backends: **MLX** (Apple GPU) or **PyTorch** (CUDA / MPS / CPU).

## Install

One extra is required and platform-specific — `mlx` (Apple Silicon) or
`torch` (everything else) — plus `viewer` (the OpenCV overlay window) and
`dev` (pytest). `uv sync` creates `.venv` and installs the project + chosen
extras in one step, from `pyproject.toml`:

**Apple Silicon:**

```sh
uv sync --extra mlx --extra viewer --extra dev
```

**CUDA / Linux / Windows / any Mac without MLX:**

```sh
uv sync --extra torch --extra viewer --extra dev
```

**NVIDIA only** — the extra above pulls a CPU-only `torch` wheel from PyPI;
install the CUDA wheel on top, or inference silently runs on CPU:

```sh
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
```
This is a `uv pip install`, not `uv sync` — re-running `uv sync` later
re-resolves `torch` back to the plain PyPI (CPU) wheel and undoes it, so redo
this line after any `uv sync --extra torch`.

**Verify the install:**

```sh
.venv/bin/python -m pytest -q
```

`.[torch]` also covers the HF-parity check (mlx port validated against the
original): `tools/compare_hf.py`, `tools/compare_hf.py --own-preproc`.

## Apple Silicon vs. everything else

**The default backend is `mlx`, unconditionally — on every platform.** There
is no auto-detection. On Linux/Windows/CUDA, or any Mac without the `mlx`
extra installed, running `segformer run` with no flags tries to import `mlx`
and fails. You must say which backend you want, every time, one of two ways:

```sh
.venv/bin/python -m segformer run --backend torch ...   # per-command
export SEGFORMER_BACKEND=torch                            # or once per shell
```
(`--backend` wins if both are set.)

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
  2. depth-cluster — B0 emits ~5 fragmented door blobs; group blobs on the
     same depth plane into one mask per physical door, so a second door at a
     different depth stays a separate detection (`geometry.cluster_doors`);
  3. depth gate (`geometry.door_gate`) — back-project, require a door-sized
     (~0.4–1.9 m wide), planar, roughly vertical structure; tolerant of a door
     only partly in frame — runs once per clustered door;
  4. temporal tracker (`tracking.py`) — 3 hits to confirm, lingers 6 frames;
     kills the frame-to-frame flicker.
- **Threaded run loop** (`runner.py`) — inference on a worker thread, `imshow`
  on the main thread. Overlay + display hide behind the next inference instead
  of adding to it. `run --no-thread` for the serial loop.

## Model management

Manifest: `segformer/models.toml`. **Three names registered today:
`segformer-b0-ade` (default), `segformer-b1-ade`, `segformer-b2-ade`** — run
`models list`, don't assume any other name works.

**Registering a model is manual, and — for the mlx backend — it's two separate
manual steps, not one.** Nothing downloads an arbitrary HF repo on its own.
`pull`/`convert`/`verify`/`path` only operate on a name already in the
manifest.

1. `tools/model_info.py <hf-repo-id>` prints a ready-to-paste `[models.<name>]`
   stanza — revision, weight file, sha256, size — and, if needed, a second
   block for step 2 below:
   ```sh
   PY=.venv/bin/python
   $PY tools/model_info.py nvidia/segformer-b3-finetuned-ade-512-512
   ```
2. Paste the `[models.<name>]` block into `models.toml`. This alone is enough
   for **`--backend torch`** — nothing else needed, skip to step 4.
3. **`--backend mlx` only:** the architecture's dimensions also have to exist
   in `segformer/model/config.py`'s `CONFIGS` dict, or `SegFormer(CONFIGS[arch])`
   raises `KeyError` — a manifest stanza alone isn't enough. Currently only
   `b0`/`b1`/`b2` are in `CONFIGS`; `b3`/`b4`/`b5` are not. `tools/model_info.py`
   detects this and prints the missing `MiTConfig(...)` block, read from the
   repo's own `config.json` — paste it in and add `"<arch>": MIT_<ARCH>` to
   `CONFIGS`.
4. `models pull <name>` then, **mlx only**, `models convert <name>` (below).

**`segformer models <subcommand>`** — every one:

| subcommand | argument | what it does |
|---|---|---|
| `list` | — | print every registered name: arch, class set, source, state. `*` marks the default (`[defaults].model` in the manifest). State is `-` (nothing local), `source` (raw file downloaded, not yet converted), or `ready` (MLX artifact built). |
| `pull` | `[name]`, optional — default from `[defaults]` | download the pinned source file into `weights/<name>/`, verify it against the manifest's `sha256`. Reuses an existing `~/.cache/huggingface` copy instead of re-fetching, if one matches. Does **not** build the MLX artifact. |
| `convert` | `[name]`, optional | **mlx only.** pull if needed, then build `weights/<name>/model.mlx.safetensors` from the source (`.bin` sources go through torch first — needs `.[torch]` installed for this step even if you'll run inference on `mlx`). The torch backend never uses this file. |
| `verify` | `[name]`, optional | re-hash the downloaded source against the manifest's `sha256`; if an MLX artifact exists, also try loading it (skipped with a one-line note if `mlx` isn't installed — doesn't fail the command). |
| `path` | `[name]`, optional | print where the MLX artifact would be / is (`weights/<name>/model.mlx.safetensors`), whether or not it's built yet. |
| `where` | — | print the resolved weights cache directory. |

```sh
PY=.venv/bin/python
$PY -m segformer models list                        # what's registered
$PY -m segformer models where                        # print the weights cache dir
$PY -m segformer models pull segformer-b2-ade        # download + sha256-verify
$PY -m segformer models convert segformer-b2-ade     # mlx only — build the artifact
$PY -m segformer models verify segformer-b2-ade       # re-check hash (+ artifact load)
$PY -m segformer models path segformer-b2-ade         # print the artifact path
```

Cache dir: `$SEGFORMER_MODELS_DIR` if set, else `<repo>/weights/` (gitignored).

`run --model <name>` / `bench --model <name>`, **`--backend mlx`**:
auto-`pull`+`convert` an already-**registered** name the first time it's used.
**`--backend torch`**: if the manifest's `file` is already `.safetensors`
(b0), it loads straight from `repo`/`revision` via `transformers`, no
`weights/` involvement. If it's a `pytorch_model.bin` (b1/b2 today), the
torch backend does its own one-time conversion into `weights/<name>/` —
sha256-pinned source download, then a local `model.safetensors` +
`config.json` it loads from on every subsequent run, never a raw `.bin` load
at inference time (`registry.ensure_torch_source`, not a `models` subcommand
— it happens automatically the first time `--backend torch` is used with
that model). Either way, no CLI command here registers a new name for you;
that's always the manual step above. A fine-tune: same stanza shape with
`source = "local"`, drop the weights file in `weights/<name>/` yourself, then
`models convert <name>`.

## Usage

`.venv` is whichever you set up in the install block above (`mlx` or `torch`).
All four commands (`probe`, `run`, `bench`, `replay`) take `--endpoint`
(default `tcp://127.0.0.1:5556`) — the ZeroMQ address to connect to (`probe`,
`run`, `bench`) or bind (`replay`).

### `probe` — sanity check, no model

```sh
PY=.venv/bin/python
$PY -m segformer probe                              # decode one frame off --endpoint, print its stats, exit
```
No other flags. Confirms a publisher is reachable and the frame contract
decodes, without loading a model.

### `run` — the main loop: consume, segment, show

```sh
PY=.venv/bin/python
$PY -m segformer run --backend mlx --overlay          # live window
$PY -m segformer run --backend torch --save out/      # headless, PNG per frame
```

| flag | default | meaning |
|---|---|---|
| `--backend {mlx,torch,dummy}` | `$SEGFORMER_BACKEND`, else `mlx` | which segmenter — see "Apple Silicon vs. everything else" above |
| `--model <name>` | `[defaults].model` in `models.toml` | which registered model — see "Model management" above |
| `--checkpoint <path-or-repo>` | none | **per-invocation override of the weights file**, skipping the registry for this one call. `--model` is still read, just for the architecture (`arch` → input size) — the checkpoint's own architecture isn't inspected, so point `--model` at a stanza with a matching `arch` or shapes won't line up. Meaning differs by backend: with `--backend mlx` it must be a path to an **already-converted** `model.mlx.safetensors` (the output of `models convert` — mlx can't take a raw repo or raw file here); with `--backend torch` it's a raw HF repo id *or* local directory, handed straight to `transformers.from_pretrained` — no manifest entry needed for this one call, but also no revision pin (you get whatever `main` currently is). |
| `--overlay` | off | open an OpenCV window: RGB + tinted door/railing masks + fps |
| `--save DIR` | none | write one PNG per frame to `DIR` instead of / as well as showing a window |
| `--max-frames N` | `0` (unlimited) | stop after `N` frames |
| `--no-thread` | off | run the old serial loop (decode→infer→draw, one thread) instead of the default worker-thread + display-thread split. Only matters when `--overlay` or `--save` is set — `run` with neither always runs the print-only serial loop. |

### `bench` — per-frame latency / fps, no display

```sh
PY=.venv/bin/python
$PY -m segformer bench --backend mlx --frames 300
```
Same `--backend` / `--model` / `--checkpoint` as `run` (identical meaning).
Plus:

| flag | default | meaning |
|---|---|---|
| `--frames N` | `300` | how many post-warmup frames to time (the very first inference is excluded — it pays one-time GPU/kernel-compile cost) |

### `replay` — publish a recorded session (no camera, no xtion repo)

`segformer replay` publishes a recorded session on ZeroMQ — indistinguishable
from a live camera to `segformer run`. This is all the capture side a
collaborator needs; no OpenNI2, no Rosetta, no `xtion` checkout.

```sh
PY=.venv/bin/python
$PY -m segformer replay session2/ --info            # frames, duration, size on disk — no publishing
$PY -m segformer replay session2/ --loop            # terminal 1: publish, original recorded timing
$PY -m segformer run --backend torch --overlay      # terminal 2: consume + segment
```

| flag | default | meaning |
|---|---|---|
| `session` | required, positional | path to a recording folder (`meta.json` + `chunk_*.npz`) |
| `--info` | off | print frame count, duration, fps, intrinsics, size on disk, and **exit** — nothing is published |
| `--fast` | off | ignore the recording's original timestamps, publish as fast as the reader can go, instead of pacing to match real-time playback |
| `--loop` | off | restart from the first frame when the recording ends, instead of stopping |
| `--compress` | off | lz4-compress the arrays **on the wire** for this publish. Independent of whether the recording itself was written with disk compression — this only affects the ZeroMQ messages `segformer run`/`probe` receive. |

Recordings are large (GB/minute) and not in git — move them out of band. Format
+ details: [`docs/recording-format.md`](docs/recording-format.md).

## Status

| part | state |
|---|---|
| `stream/` — frame contract, codec, recording, replay, subscribe | done — independent of the net |
| `segmenter.py`, `overlay.py` | done |
| `model/` — MiT encoder + decode head (MLX) | done; `tools/compare_hf.py` currently shows ~5e-3 mean logit diff / 99.97% argmax agreement vs HF, not bit-exact — see "Known limits" |
| `model/registry.py` + `models.toml` | done — pinned, sha256, offline-first; b1/b2 also converted to a local safetensors copy for the torch backend (see "Model management") |
| `model/convert.py` — HF safetensors → MLX artifact | done |
| `backends/postprocess.py` | done — shared prob→mask→gate→tracker pipeline |
| `backends/mlx_backend.py` | done — MLX SegFormer → probability maps, fp16 weights by default |
| `backends/torch_backend.py` | done — HF transformers on CUDA / MPS / CPU → probability maps, fp16 weights on GPU (CUDA/MPS), fp32 on CPU |
| `geometry.py` — depth cluster + door gate | done, tuned on session2 (one real door); `cluster_doors` now emits one mask per depth-separated door instead of capping at one door per frame |
| `tracking.py` — temporal hysteresis | done |
| `runner.py` — threaded run loop | done |
| `bench` command | done |
| perf (session2, this Mac, 150 frames) — see table below | ok |

`bench --frames 150` on an M-series Mac, mixed door/no-door frames:

| model | backend | mean latency | fps |
|---|---|---|---|
| b0 | mlx | 37.7 ms | 26.5 |
| b0 | torch (MPS, fp16) | 32.2 ms | 31.1 |
| b2 | mlx | 78.9 ms | 12.7 |
| b2 | torch (MPS, fp16) | 63.2 ms | 15.8 |

**`torch` is currently faster than `mlx` on this same Apple Silicon machine**,
at both sizes, despite `mlx` being the "primary" backend. Both already run
fp16 weights by default here — that's not the gap. Two real differences in
the MLX port instead: (1) `SegFormer.class_probs()`
(`segformer/model/segformer.py`) calls `mx.eval()` to pull the low-res
probability map to the CPU, *then* upsamples to frame size with `cv2.resize`
— the torch backend keeps the upsample on-GPU (`F.interpolate`) and only
syncs once, after; (2) the MLX port never calls `mx.compile` anywhere, so
every attention/MLP block dispatches as separate unfused Metal kernels, where
`torch`'s MPS backend has fused kernels for the same ops built in. Neither is
fixed yet — `mlx` stays the default backend regardless, so the two ports stay
comparable when someone eventually needs to run this on iOS/embedded MLX
rather than a MPS Mac.

## Known limits

- **MLX-vs-HF parity is not currently bit-exact** — `tools/compare_hf.py
  --own-preproc` shows cos-sim 1.0, ~99.97% argmax agreement, but ~5e-3 mean
  logit diff, above its own `<1e-3` bar. Same magnitude on b0/b1/b2 alike, and
  the pinned revision matches HF's current `main` exactly, so it isn't a
  weight/revision problem — most likely drift between whatever `transformers`
  version this was last validated against and the one installed now
  (`transformers` 5.16.1 / `torch` 2.14 here). Functionally fine (99.97%
  agreement), just don't take the old "bit-exact" claim at face value.
- **B0 (the default) is only ~37 mIoU** and has no real advantage over b1/b2
  anymore now that the torch backend runs fp16 too — it exists mainly because
  it's the one model with a native HF safetensors file, needing no local
  conversion (see "Model management"). Precision on doors comes from the
  depth gate, not the model, at any of these sizes. b3/b4/b5 need the manual
  "Model management" steps first — the `.bin` → MLX conversion itself is done
  (`model/convert.py`), it's the registry + `CONFIGS` entries that are
  missing.
- **One real door, tuned on it** — session2 has a single door (short USB
  tether, door barely fits the frame). On a 250-frame segment with it in view:
  detected ~96% of frames, 2 on/off flips. session1 (no door): 0 false doors.
  Multi-door footage now exists (mobile recordings), but hasn't been run
  through this pipeline yet to confirm `cluster_doors` behaves on it — see
  HANDOFF.md.
- **b1/b2 only ship `pytorch_model.bin` on HF** (checked directly — no
  `refs/pr/2`, no safetensors variant at all, on either repo). The torch
  backend never repeatedly unpickles that file: `registry.ensure_torch_source`
  converts it to a local, sha256-source-pinned `model.safetensors` +
  `config.json` once, and `from_pretrained` loads that local directory from
  then on — still going through transformers' own old→new key renaming
  (needed: `transformers` 5.x moved SegFormer's internal module names, a raw
  `load_state_dict` would silently produce a randomly-initialized model
  instead of erroring).

## Next

1. Railing: B0 barely sees railings (2 blobs in all of session2). Needs a
   bigger model or a depth-primary detector (thin repeated vertical structure).
2. Run the new mobile multi-door recordings through `cluster_doors` / the full
   pipeline — it's only been checked against synthetic depth so far
   (`tests/test_geometry.py`).
3. Split door-from-wall when B0 merges them beyond depth-clustering (a
   depth-plane discontinuity within the merged region).
4. mlx catch-up: `mx.compile` the forward pass, keep the probability-map
   upsample on-GPU instead of syncing early — see the perf table above.

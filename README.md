# Help for Visually Impaired People — PoC

<img src="figures/detect_export.png" alt="A door detected and outlined, labelled &quot;door 1.34&quot;" width="320" align="left" style="margin: 0 16px 16px 0">

This is a proof of concept for a navigation aid for blind and visually
impaired people. A camera watches the room. The tool finds **doors,
railings, and stairs** in that video, then turns each one into a **sound**
— a tone that comes from the right direction, at the right volume for how
far away it is. No screen, no braille, just sound: the way a blind user
could actually use it.

The video comes from the [`xtion`](../xtion) camera pipeline. The detection
model is **SegFormer** (Xie et al., NeurIPS 2021), pretrained on **ADE20K**
— it already knows `door`, `railing`, `bannister`, `screen door`, `stairs`,
and `stairway`, so no extra training was needed. Two backends run the
model: **MLX** (Apple GPU) or **PyTorch** (CUDA / MPS / CPU).

<br clear="left">

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
                   segformer run  ─▶ wire.decode ─▶ segmenter ─▶ tracker ─▶ ┬─▶ overlay   (--overlay)
                   tcp://127.0.0.1:5556  SegFormer B0/B1/B2, ADE20K         └─▶ consumer/  (spatial audio, on by default)
                   (mlx | torch backend) → door/railing/stairs masks → depth gate (door/railing only)
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
  probability→mask→tracker pipeline after the network (`backends/postprocess.py`,
  see below); the backends differ only in producing the probability maps.
  Pick with `--backend`, `$SEGFORMER_BACKEND`, else `mlx`.
- **Precision stack** (B0 is only 37 mIoU, so filtering does the work). Door
  and screen-door only get steps 2–3 (`labels.DOORLIKE`) — railing, bannister
  and stairs go straight from step 1 to the tracker, no depth gate:
  1. threshold the *probability* map — 0.85 for door/screen-door (argmax eats
     the wall around a door), 0.45 for everything else;
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
$PY -m segformer run --backend torch                                      # just this: spatial audio, on by default
$PY -m segformer run --backend torch --overlay --model segformer-b1-ade   # + RGB window, non-default model
$PY -m segformer run --backend torch --save out/                          # headless, PNG per frame
```

| flag | default | meaning |
|---|---|---|
| `--backend {mlx,torch,dummy}` | `$SEGFORMER_BACKEND`, else `mlx` | which segmenter — see "Apple Silicon vs. everything else" above |
| `--model <name>` | `[defaults].model` in `models.toml` | which registered model — see "Model management" above |
| `--checkpoint <path-or-repo>` | none | **per-invocation override of the weights file**, skipping the registry for this one call. `--model` is still read, just for the architecture (`arch` → input size) — the checkpoint's own architecture isn't inspected, so point `--model` at a stanza with a matching `arch` or shapes won't line up. Meaning differs by backend: with `--backend mlx` it must be a path to an **already-converted** `model.mlx.safetensors` (the output of `models convert` — mlx can't take a raw repo or raw file here); with `--backend torch` it's a raw HF repo id *or* local directory, handed straight to `transformers.from_pretrained` — no manifest entry needed for this one call, but also no revision pin (you get whatever `main` currently is). |
| `--overlay` | off | open an OpenCV window: RGB + tinted door/railing/stairs masks + fps |
| `--save DIR` | none | write one PNG per frame to `DIR` instead of / as well as showing a window |
| `--no-spatial` | off (spatial **on** by default) | skip the spatial-audio consumer -- the radar window and its one-tone-per-class audio. See "Spatial view" below. |
| `--mute` | off | keep the radar window, skip its audio |
| `--record PATH` | none | write the overlay + spatial windows, side by side, to one video file at `PATH` — with sound, unless `--mute`. Needs both windows (`--overlay` on, spatial on — the default). See "Recording" below. |
| `--record-fps N` | `15` | playback rate for `--record` — a fixed rate, not the measured inference fps |
| `--max-frames N` | `0` (unlimited) | stop after `N` frames |
| `--no-thread` | off | run the old serial loop (decode→infer→draw, one thread) instead of the default worker-thread + display-thread split. `run` with no flags at all still opens the spatial window (on by default) and runs threaded; pass `--no-spatial` too for the old bare print-only serial loop. |

#### Spatial view

`consumer/`, apart from the net. This is the actual point of the project, so
it's on by default -- `--no-spatial` skips it, `--mute` keeps the window but
drops its audio.

- **Left:** top-down radar. You are at the bottom. Rings are 1 m apart. Lines are the camera's field of view.
- **Right:** one row per object. Bearing, distance, and a bar split into left and right ear. Louder means closer.
- **Distance** is the median depth of the mask interior. The network never sees depth.
- **No depth reading:** the dot sits on the radar edge as a hollow ring. Bearing is still right. Distance shows `?`.
- Bearing and distance are smoothed per object id.
- **Audio:** one continuous tone per class -- door 220 Hz, railing 440 Hz, stairs
  660 Hz (both door types share a tone, same for railing/bannister). Panned with
  both volume and a small interaural delay. Needs `sounddevice` (`pip install
  'segformer[audio]'`); missing it, or `--mute`, just runs the radar silent.
- Press `1`/`2`/`3` in the spatial window to mute/unmute door/railing/stairs.
  Confirmed by a chirp (one falling pip = muted, two rising = unmuted) — the
  end user is blind, so that chirp is the real feedback; the HUD on the
  radar panel showing the same state is only for a sighted dev at a terminal.
- Press `q` in either window to quit.

#### Recording (`--record PATH`)

```sh
PY=.venv/bin/python
$PY -m segformer run --backend torch --overlay --record out/demo.mp4
```

Writes the overlay window and the spatial window side by side into one
video file. The sound is real: `consumer/audio.py` taps the exact samples
it hands to the speaker (not a system-audio capture) and writes them to a
temp WAV alongside a temp silent video; both get muxed together with
`ffmpeg` into `PATH` when the run ends, then cleaned up. Mute toggles and
chirps are in the recording, since they're part of that same signal.

No `ffmpeg` on the machine, or `--mute`, or the audio device never opened:
the output is still written, just as video with no audio track — `run`
says so when it finishes.

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
| `model/` — MiT encoder + decode head (MLX) | done; not bit-exact vs HF — see "Known limits" |
| `model/registry.py` + `models.toml` | done — pinned, sha256, offline-first; b1/b2 also converted to a local safetensors copy for the torch backend (see "Model management") |
| `model/convert.py` — HF safetensors → MLX artifact | done |
| `backends/postprocess.py` | done — shared prob→mask→tracker pipeline, gate only for door/screen-door |
| `backends/mlx_backend.py` | done — MLX SegFormer → probability maps, fp16 weights by default |
| `backends/torch_backend.py` | done — HF transformers on CUDA / MPS / CPU → probability maps, fp16 weights on GPU (CUDA/MPS), fp32 on CPU |
| `geometry.py` — depth cluster + door gate | done, tuned on session2 (one real door); `cluster_doors` now emits one mask per depth-separated door instead of capping at one door per frame |
| `tracking.py` — temporal hysteresis | done |
| `runner.py` — threaded run loop | done |
| `consumer/spatial.py` — spatial cues (direction/distance), on by default | done. Draws the radar. Checked on synthetic depth and one real recording. |
| `consumer/audio.py` — spatial audio for the cues | done. One tone per class (door/railing/stairs), panned left/right by direction. Press 1/2/3 to mute one; a short sound confirms it, since the user can't see a screen. |
| `segformer/recorder.py` — `--record`, overlay+spatial to one video with sound | done. Muxed with `ffmpeg`; falls back to video-only if it's not installed. |
| `bench` command | done |
| perf (session2, Mac Studio M1 Max, 150 frames) — see table below | ok |

`bench --frames 150` on a Mac Studio (Apple M1 Max, 32 GB), mixed door/no-door frames:

| model | backend | mean latency | fps |
|---|---|---|---|
| b0 | mlx | 37.7 ms | 26.5 |
| b0 | torch (MPS, fp16) | 32.2 ms | 31.1 |
| b2 | mlx | 78.9 ms | 12.7 |
| b2 | torch (MPS, fp16) | 63.2 ms | 15.8 |

**`torch` is currently faster than `mlx` on this same machine**, at both
sizes — see the table above. `mlx` stays the default backend anyway, for
when this needs to run on iOS or embedded MLX later.

## Known limits

- **MLX output does not exactly match the original HF model.**
  `tools/compare_hf.py --own-preproc` checks this. It shows 99.97% agreement
  on the final answer (argmax), but the raw numbers (logits) differ by about
  5e-3 on average — the target was under 1e-3. Same gap on b0, b1, and b2.
  The pinned model revision matches HF's current version exactly, so that's
  not the cause. The real cause is not confirmed. In practice this is fine
  — 99.97% agreement is high. Just don't trust an old claim that it was
  "bit-exact".
- **B0 is the default model, but it is not very accurate** — only about 37
  mIoU on ADE20K. It's the default because it's the only model with a native
  HF safetensors file, not because it detects better than b1 or b2. b1 and
  b2 haven't been measured here, so we can't say which is more accurate.
  Door detection stays accurate anyway, because the depth gate — not the
  model — does that work, at any model size. b3, b4, and b5 aren't set up
  yet. The code to convert them to MLX already exists (`model/convert.py`).
  What's missing is adding them to the registry and to `CONFIGS` — see
  "Model management".
- **b1 and b2 only have an old-style weights file (`pytorch_model.bin`) on
  HuggingFace.** We checked directly — no safetensors version exists for
  either model. The torch backend doesn't reload that file every time. The
  first time you use `--backend torch` with b1 or b2, it converts the file
  once into a local `model.safetensors` + `config.json`
  (`registry.ensure_torch_source`). Every run after that loads the local
  file instead. Loading still goes through `transformers`' own renaming
  step — `transformers` 5.x renamed SegFormer's internal module names.
  Skipping that step would silently load a broken, randomly-initialized
  model, with no error.

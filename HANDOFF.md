# Handoff — segformer + xtion

The MacBook + Xtion can record **many doors + railings** (the old blocker
— Mac Studio's short USB tether meaning one door, barely in frame — is gone).
`cluster_doors` (below) groups blobs by depth plane, one mask per physical
door, for exactly this case.

## Where it stands

Two repos, `../xtion` (capture) and this one (segmentation). Both committed clean.

- **xtion**: RGB-D capture over ZMQ, schema-v1 frame contract. `capture serve
  --device xtion` runs the camera (x86_64 CPython under Rosetta — libOpenNI2 is
  x86_64). `capture replay session2/ --publish --loop` feeds recordings.
- **segformer**: SegFormer (ADE20K) ported to MLX + a torch backend (HF
  transformers). `segformer run --backend mlx --overlay` consumes the stream,
  segments door/railing, overlays. Model registry: `segformer models …`.
  b0/b1/b2 all registered and working on both backends — b0 no longer has a
  real edge over b1/b2 now that torch also runs fp16 (see "Backends" below);
  it's kept mainly because it's the one model with a native HF safetensors
  file.
- MLX-vs-HF parity (`tools/compare_hf.py`) is **not currently bit-exact** —
  cos-sim 1.0 and ~99.97% argmax agreement, but ~5e-3 mean logit diff, above
  the tool's own `<1e-3` bar, on all of b0/b1/b2 alike (so not a b1/b2-specific
  regression). Traced to a `transformers`/`torch` version mismatch against
  whatever was installed when this was last "bit-exact" (currently
  `transformers` 5.16.1 / `torch` 2.14 — ruled out eager-vs-sdpa attention as
  the cause, didn't dig further). Not urgent — 99.97% argmax agreement is
  functionally fine — but don't trust the old "bit-exact" claim at face value.

## Door pipeline — works on the one door we have

session2 (one real door): ~96% detection when in view, flicker gone.
session1 (no door): 0 false positives.

`cluster_doors` emits one mask per depth-separated blob group instead of
collapsing everything onto the single biggest blob (the old
one-door-per-frame cap).

Precision comes from filtering, not the model (all sizes are shared here, B0
is only 37 mIoU):
1. threshold door **probability** > 0.85 (not argmax) — `backends/postprocess.py: PROB_THRESH`
2. depth-cluster B0's fragmented blobs into one mask per door — `geometry.cluster_doors()`
3. depth gate each clustered door — `geometry.door_gate()` (size / planar / vertical)
4. temporal tracker — `segformer/tracking.py` (3 hits confirm, 6 linger)

All four were tuned against **one door**. Expect to re-tune constants once
there's a real multi-door dataset:
- `geometry.py`: `DOOR_W`, `DOOR_H_MIN/MAX`, `WALL_W`, `PLANE_INLIER`, `MAX_NORMAL_Y`
- `backends/postprocess.py`: `PROB_THRESH`, `MIN_COMP_PX`, `min_area` (passed in from the CLI)
- `geometry.py` clustering: `DEPTH_MERGE_TOL`, `MERGE_CLOSE_PX`
- `tracking.py`: `min_hits`, `max_misses`, `iou_match`

## Next steps

1. Label a handful of frames (even rough boxes) → a small eval set.
   Add `tools/eval.py`: precision/recall of door/railing detections vs labels.
2. Re-tune the constants below against the eval set, not by eye.
3. **Railings**: B0 barely sees them (2 blobs in all of session2). b1/b2 are
   registered and confirmed working now (fp16 fixed the b2 slowness) — check
   whether they see railings better before reaching for a depth-primary
   detector (thin repeated vertical structure at hand height, lots of
   depth-gradient / see-through gaps).
4. Live fps: threaded run loop is in `segformer/runner.py` (infer worker +
   main-thread display). See "Backends" below for current per-model/backend
   numbers — `torch` is faster than `mlx` on this Mac right now, worth using
   for anything latency-sensitive until the mlx-side gaps are closed.

## Backends

- `--backend mlx | torch | dummy`, else `$SEGFORMER_BACKEND`, else `mlx`.
- **mlx** — Apple GPU, the hand-port, `.[mlx]`. **torch** — HF transformers on
  CUDA / MPS / CPU, `.[torch]`, for a machine without MLX. Both feed the same
  `backends/postprocess.py` (threshold → components → depth gate → tracker);
  they differ only in producing the probability maps. Both run fp16 weights
  on GPU by default now (mlx always did; torch backend does too as of the
  fp16 cast added in `torch_backend.load()`, fp32 only when the device is CPU).
- **`torch` is currently *faster* than `mlx` on this same Apple Silicon Mac**
  — `bench --frames 150` on session2: b0 mlx 37.7ms/26.5fps vs torch 32.2ms/
  31.1fps; b2 mlx 78.9ms/12.7fps vs torch 63.2ms/15.8fps. fp16 isn't the
  differentiator (both have it) — two real gaps in the MLX port: (1)
  `SegFormer.class_probs()` (`segformer/model/segformer.py`) does `mx.eval()`
  to pull the low-res probability map to CPU *before* upsampling with
  `cv2.resize`, where torch upsamples on-GPU (`F.interpolate`) and only syncs
  once, at the very end; (2) the MLX port never calls `mx.compile` — every
  attention/MLP block is separate unfused Metal kernel launches, vs. torch's
  MPS backend having fused kernels for the same ops. Neither fixed yet.
  `mlx` stays the default backend anyway.
- b1/b2 only ship `pytorch_model.bin` on HF (checked: no `refs/pr/2`, no
  safetensors variant on either repo). The torch backend now converts that to
  a local safetensors copy once (`registry.ensure_torch_source`) instead of
  unpickling it on every load — see README "Model management".
- `segformer/labels.py` is framework-free; `segformer/model/__init__.py` is
  lazy — a `.[torch]`-only install never imports mlx.

## Spatial-audio mock (branch `spatial-consumer-mvp`)

`segformer run --spatial` opens a second window. It mocks the downstream consumer.

- Code: `consumer/spatial.py`. Tests: `tests/test_spatial.py`.
- Per object: bearing from the mask, distance from the median depth of the eroded mask.
- Shows a top-down radar and a left/right pan and loudness bar. No audio is played.
- Distance is unknown (`?`) when the mask has no depth. Bearing is still shown.
- Only checked on synthetic depth and one recording (`session_rappu`, two railings). No door footage yet.
- A long object such as a railing gets one point (the median). Fine for an MVP.

## Working from recordings

`segformer replay <session>/ --loop` publishes a recorded session on ZMQ — a
collaborator needs only this repo (`.[torch,viewer,dev]`) + the session files,
no `xtion` checkout / OpenNI2 / Rosetta. `stream/replay.py` + `stream.wire.encode_frame`.
Recordings are GB-scale, not in git — transfer out of band.

## Gotchas

- xtion capture MUST run on the x86_64 venv (`xtion/.venv`). arm64 = no libOpenNI2.
- segformer venv: `.[mlx]` on Apple Silicon, `.[torch]` elsewhere. Both + `viewer,dev`.
- `tools/compare_hf.py` needs `.[torch]` — MLX-vs-HF parity check only.
- recordings are ~44 MB/s uncompressed; session2 is 2.5 GB. gitignored.

# Handoff — paused pending mobile recordings

Paused until the Xtion can run off a MacBook and record **many doors + railings**
(short USB tether on the Mac Studio = one door, barely in frame).

## Where it stands

Two repos, `../xtion` (capture) and this one (segmentation). Both committed clean.

- **xtion**: RGB-D capture over ZMQ, schema-v1 frame contract. `capture serve
  --device xtion` runs the camera (x86_64 CPython under Rosetta — libOpenNI2 is
  x86_64). `capture replay session2/ --publish --loop` feeds recordings.
- **segformer**: SegFormer-B0 (ADE20K) ported to MLX, validated bit-exact vs HF.
  `segformer run --backend mlx --overlay` consumes the stream, segments
  door/railing, overlays. Model registry: `segformer models …`.

## Door pipeline — works on the one door we have

session2 (one real door): ~96% detection when in view, flicker gone.
session1 (no door): 0 false positives.

Precision comes from filtering, not the model (B0 is 37 mIoU):
1. threshold door **probability** > 0.85 (not argmax) — `MlxSegmenter._PROB_THRESH`
2. depth-consolidate B0's fragmented blobs — `geometry.consolidate_door()`
3. depth gate on the consolidated region — `geometry.door_gate()` (size / planar / vertical)
4. temporal tracker — `segformer/tracking.py` (3 hits confirm, 6 linger)

All three were tuned against **one door**. Expect to re-tune constants once
there's a real dataset:
- `geometry.py`: `DOOR_W`, `DOOR_H_MIN/MAX`, `WALL_W`, `PLANE_INLIER`, `MAX_NORMAL_Y`
- `mlx_backend.py`: `_PROB_THRESH`, `min_area`, `_MIN_COMP`
- `geometry.py` consolidation: `DEPTH_MERGE_TOL`, `MERGE_CLOSE_PX`
- `tracking.py`: `min_hits`, `max_misses`, `iou_match`

## When recordings exist — next steps

1. Record varied doors (open/closed, wide/narrow, angled, partial) + railings.
   `capture record --device xtion --seconds N --out sessionN/`.
2. Label a handful of frames (even rough boxes) → a small eval set.
   Add `tools/eval.py`: precision/recall of door/railing detections vs labels.
3. Re-tune the constants above against the eval set, not by eye.
4. **Railings**: B0 barely sees them (2 blobs in all of session2). Options:
   - bigger model — b1/b2 already registered, try those first. b3-b5 need
     `tools/model_info.py <repo>` (prints both the `models.toml` stanza *and*,
     for mlx, the missing `MiTConfig` for `segformer/model/config.py` — see
     README "Model management").
   - or a depth-primary railing detector: thin repeated vertical structure at
     hand height, lots of depth-gradient / see-through gaps.
5. Live fps: threaded run loop is in `segformer/runner.py` (infer worker +
   main-thread display), ~26 fps effective. Model forward (~22 ms) is the
   floor. For more: smaller input (384^2) or a bigger+quantised model.

## Backends

- `--backend mlx | torch | dummy`, else `$SEGFORMER_BACKEND`, else `mlx`.
- **mlx** — Apple GPU, the hand-port, `.[mlx]`. **torch** — HF transformers on
  CUDA / MPS / CPU, `.[torch]`, for a machine without MLX. Both feed the same
  `backends/postprocess.py` (threshold → components → depth gate → tracker);
  they differ only in producing the probability maps. ~equal speed on this Mac
  (mlx 27 fps, torch-MPS 26 fps).
- `segformer/labels.py` is framework-free; `segformer/model/__init__.py` is
  lazy — a `.[torch]`-only install never imports mlx.

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

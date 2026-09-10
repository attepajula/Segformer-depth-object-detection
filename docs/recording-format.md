# Recording format & replay

A recorded session is a folder the capture service's recorder writes. `segformer
replay` reads it and publishes the frames on ZeroMQ — the same stream a live
camera produces, so nothing downstream can tell the difference. No camera,
OpenNI2, or Rosetta needed on the replay machine.

## On disk

```
session2/
  meta.json           schema_version, device, depth_scale, registered,
                      intrinsics {fx, fy, cx, cy, width, height},
                      chunk_frames, compressed, frame_count
  chunk_00000.npz     rgb          (N, H, W, 3) uint8   — R,G,B
                      depth        (N, H, W)    uint16  — millimetres, 0 = no reading
                      frame_id     (N,)         int64
                      timestamp_ns (N,)         int64   — monotonic capture clock
  chunk_00001.npz     ...
```

`N` = `chunk_frames` (30 by default = 1 s at 30 fps). `.npz` is plain
`numpy.savez`; `--compress` recordings use `savez_compressed` and load
transparently. ~1.5 MB/frame uncompressed → ~44 MB/s, ~2.6 GB per 60 s session.

## On the wire

One msgpack map per frame (single-part, so ZeroMQ `CONFLATE` keeps only the
newest). Contract fields + `compression` (`"none"` | `"lz4"`) + `rgb` / `depth`
as `{dtype, shape, data}` where `data` is the raw C-contiguous buffer. `device`
becomes `replay:<original>`. Guarded by `schema_version` — a mismatched frame is
rejected, not misread. Decode reference: `stream/wire.py`.

## Use

```sh
segformer replay session2/ --info                     # frames, duration, size
segformer replay session2/ --loop                     # publish, original timing
segformer replay session2/ --loop --fast              # ignore timing, go flat out
segformer replay session2/ --loop --compress          # lz4 on the wire
```
then, in another terminal:
```sh
segformer run --backend torch --overlay               # or --backend mlx
```

## Getting the recordings

The `.npz` files are large (GB per minute) and are **not** in git. Move them
out of band — a shared drive, `rsync`, or a cloud bucket. Or record shorter /
`--compress`d sessions from the capture side.

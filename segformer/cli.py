"""``segformer`` CLI.

    segformer replay <session>  [--fast] [--loop] [--info]   publish a recording
    segformer probe  [--endpoint ...]                 decode one frame, print stats
    segformer run    [--backend ...] [--overlay | --save DIR] [--max-frames N]
    segformer bench  [--backend ...] [--frames N]     per-frame latency / fps
    segformer models list|pull|convert|verify|path|where

Backend is mlx | torch | dummy — from --backend, else $SEGFORMER_BACKEND,
else mlx.
"""

from __future__ import annotations

import argparse
import sys


def cmd_probe(args) -> int:
    import zmq

    from stream.wire import decode_frame

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(args.endpoint)
    try:
        buf = sock.recv()
    except zmq.Again:
        print(f"no frame within timeout on {args.endpoint}")
        return 1
    f = decode_frame(buf)
    print(f"frame {f.frame_id} @ {f.timestamp_ns} ns  device={f.device}")
    print(f"  rgb {f.rgb.shape} {f.rgb.dtype}   depth {f.depth.shape} {f.depth.dtype}")
    print(f"  intrinsics {f.intrinsics}")
    print(f"  valid-depth pixels: {int(f.depth_valid_mask.sum())} / {f.depth.size}")
    return 0


def cmd_run(args) -> int:
    from segformer.backends import make_segmenter
    from stream.consumer import FrameConsumer, RateMeter
    from segformer import overlay

    seg = make_segmenter(args.backend, model=args.model)
    seg.load(args.checkpoint)

    args.spatial = not args.no_spatial    # on by default -- see "--no-spatial" help

    if args.record and not (args.overlay and args.spatial):
        print("  --record needs both --overlay and spatial on (spatial is on by "
              "default -- don't pass --no-spatial) -- nothing to put side by side "
              "otherwise")
        return 1

    has_display = args.overlay or args.save or args.spatial
    if has_display and not args.no_thread:
        from segformer.runner import run_threaded

        print(f"segformer run (threaded): backend={seg.name} endpoint={args.endpoint} "
              f"overlay={args.overlay} save={args.save or '-'} spatial={args.spatial} "
              f"record={args.record or '-'}")
        run_threaded(args.endpoint, seg, show=args.overlay, save_dir=args.save,
                     max_frames=args.max_frames, spatial=args.spatial, mute=args.mute,
                     record=args.record, record_fps=args.record_fps)
        return 0

    record_video_tmp = record_audio_tmp = None
    if args.record:
        import pathlib

        from segformer.recorder import temp_paths

        pathlib.Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        record_video_tmp, record_audio_tmp = temp_paths(args.record)

    if args.spatial:
        from consumer import spatial as spatial_view

        smoother = spatial_view.CueSmoother()
        audio = None
        if not args.mute:
            from consumer.audio import AudioRenderer

            audio = AudioRenderer()
            if audio.start():
                print("  spatial audio: keys 1/2/3 in the spatial window toggle "
                      "door/railing/stairs sound, q quits")

    consumer = FrameConsumer(args.endpoint)
    meter = RateMeter()
    state = None
    prev_id = None
    n = 0
    save_dir = None
    if args.save:
        import pathlib

        save_dir = pathlib.Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)

    recorder = None
    recording_begun = False   # True once the first real frame starts both tracks together
    if args.record:
        from segformer.recorder import WindowRecorder

        recorder = WindowRecorder(record_video_tmp, fps=args.record_fps)
        recorder.start()
        print(f"  recording overlay + spatial to {args.record} "
              f"({args.record_fps:.0f} fps{'' if audio is not None else ', no audio -- muted'})")

    print(f"segformer run: backend={seg.name} endpoint={args.endpoint} "
          f"overlay={args.overlay} save={args.save or '-'}")
    try:
        for frame in consumer.frames():
            if frame is None:
                print("  waiting for frames...")
                continue
            if prev_id is not None and frame.frame_id <= prev_id:
                seg.reset()          # stream restarted (replay loop / reopen)
                state = None
                if args.spatial:
                    # new instance ids start from 1 again (seg.reset() above
                    # resets the tracker) -- without this, the smoother keeps
                    # the previous loop's id=1 azimuth/distance and heavily
                    # smooths the new detection toward it, a visible and
                    # audible jump right at the loop point
                    smoother.reset()
            prev_id = frame.frame_id

            result, state = seg.infer(frame, state)
            fps = meter.tick()

            # one shared key poll per tick -- cv2's key queue is global, not
            # per-window; two independent waitKey calls (overlay + spatial)
            # would split the same stream and each window would miss keys.
            key = overlay.poll_key(1) if (args.overlay or args.spatial) else None

            if args.overlay or save_dir is not None:
                img = overlay.render(frame, result, fps=fps)
                if save_dir is not None:
                    _write_png(save_dir / f"{frame.frame_id:06d}.png", img)
                if args.overlay and not overlay.show(img, key=key):
                    break
            elif not args.spatial:
                print(f"  frame {frame.frame_id}  inst={len(result.instances)}  {fps:4.1f} fps")
            if args.spatial:
                cues = smoother.update(spatial_view.compute_cues(frame, result))
                muted = frozenset()
                if audio is not None:
                    audio.update(cues)
                    muted = audio.muted_voices()
                radar = spatial_view.render(cues, frame.intrinsics, muted=muted)
                if recorder is not None:
                    if not recording_begun:
                        # this frame is both tracks' zero point -- see
                        # AudioRenderer.begin_recording's docstring for why
                        # it can't just start with audio.start() instead
                        if audio is not None:
                            audio.begin_recording(record_audio_tmp)
                        recording_begun = True
                    recorder.update(img, radar)
                if not spatial_view.show(radar, audio=audio, key=key):
                    break

            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        seg.close()
        if args.spatial and audio is not None:
            audio.stop()                          # finishes the WAV file, if recording
        if recorder is not None:
            recorder.close()                      # finishes the silent video file
            from segformer.recorder import mux

            has_sound = mux(record_video_tmp, record_audio_tmp, args.record)
            print(f"  wrote {args.record}" + ("" if has_sound else " (no audio track)"))
    print(f"processed {n} frames")
    return 0


def _write_png(path, rgb) -> None:
    import cv2

    cv2.imwrite(str(path), rgb[:, :, ::-1])


def cmd_bench(args) -> int:
    import time

    from segformer.backends import make_segmenter
    from stream.consumer import FrameConsumer

    seg = make_segmenter(args.backend, model=args.model)
    t = time.time()
    seg.load(args.checkpoint)
    print(f"load {time.time() - t:.1f}s")

    consumer = FrameConsumer(args.endpoint)
    f = None
    while f is None:
        f = consumer.recv()
    seg.infer(f, None)  # warmup: first infer compiles Metal kernels

    durs: list[float] = []
    state = None
    while len(durs) < args.frames:
        f = consumer.recv()
        if f is None:
            continue
        t0 = time.perf_counter()
        _, state = seg.infer(f, state)
        durs.append(time.perf_counter() - t0)
    consumer.close()
    seg.close()

    d = sorted(durs)
    n = len(d)
    print(f"{seg.name}: {n} frames  mean {1000 * sum(d) / n:.1f} ms "
          f"({n / sum(d):.1f} fps)  p50 {1000 * d[n // 2]:.1f}  "
          f"p90 {1000 * d[int(n * 0.9)]:.1f}  max {1000 * d[-1]:.1f}")
    return 0


def cmd_replay(args) -> int:
    from stream import replay

    if args.info:
        return replay.info(args.session)
    return replay.publish(args.session, args.endpoint, fast=args.fast, loop=args.loop,
                          compression="lz4" if args.compress else "none")


def cmd_models(args) -> int:
    from segformer.model import registry as reg

    models = reg.manifest()

    if args.models_cmd == "where":
        print(reg.models_root())
        return 0

    if args.models_cmd == "list":
        default = reg.default_model()
        for name, spec in models.items():
            art = reg.artifact_path(spec)
            src = reg.source_path(spec)
            state = "ready" if art.is_file() else ("source" if src.is_file() else "-")
            mark = "*" if name == default else " "
            print(f"{mark} {name:22s} {spec.arch:3s} {spec.classes:16s} "
                  f"{spec.source:5s} [{state}]")
        return 0

    spec = reg.get(args.name)
    if args.models_cmd == "path":
        print(reg.artifact_path(spec))
        return 0
    if args.models_cmd == "pull":
        p = reg.ensure_source(spec, quiet=False)
        print(f"source: {p}  ({p.stat().st_size} bytes, sha256 ok)")
        return 0
    if args.models_cmd == "convert":
        p = reg.ensure_artifact(spec, quiet=False)
        print(f"artifact: {p}")
        return 0
    if args.models_cmd == "verify":
        src = reg.source_path(spec)
        if not src.is_file():
            print(f"{spec.name}: source not downloaded (run: segformer models pull {spec.name})")
            return 1
        got = reg.sha256_file(src)
        ok = (got == spec.sha256) if spec.sha256 else None
        print(f"{spec.name}: source sha256 {got} "
              f"{'OK' if ok else 'MISMATCH' if ok is False else '(no pin)'}")
        art = reg.artifact_path(spec)
        if art.is_file():
            try:
                import mlx.core as mx

                from segformer.model import load_model

                load_model(spec.name)
                print(f"  artifact loads OK ({len(mx.load(str(art)))} tensors)")
            except ImportError:
                print("  (mlx not installed — skipping artifact load check)")
        return 0 if ok is not False else 1
    return 2


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--endpoint", default="tcp://127.0.0.1:5556")

    p = argparse.ArgumentParser(prog="segformer", description=__doc__, parents=[common],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", parents=[common], help="subscribe, decode one frame, print stats")
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("run", parents=[common], help="consume + segment + overlay")
    sp.add_argument("--backend", default=None, choices=["mlx", "torch", "torch-mps", "dummy"],
                    help="default: $SEGFORMER_BACKEND, else mlx")
    sp.add_argument("--model", default=None,
                    help="registry name from models.toml (default from [defaults])")
    sp.add_argument("--checkpoint", default=None,
                    help="override: a local weights file, or (torch backend) an HF repo id -- "
                         "skips the registry entirely")
    sp.add_argument("--overlay", action="store_true", help="OpenCV window")
    sp.add_argument("--save", default=None, metavar="DIR", help="write overlay PNGs")
    sp.add_argument("--no-spatial", action="store_true",
                    help="skip the spatial-audio consumer -- on by default: a second window "
                         "(top-down radar + L/R pan) and its audio -- one tone for both door "
                         "types, one for railing/bannister, one for stairs -- unless --mute. "
                         "1/2/3 in that window mute/unmute a class, confirmed by a chirp.")
    sp.add_argument("--mute", action="store_true",
                    help="keep the radar window, skip its audio output")
    sp.add_argument("--record", default=None, metavar="PATH",
                    help="write overlay + spatial windows side by side to one video "
                         "file. Needs both --overlay and spatial on (spatial is "
                         "already on by default).")
    sp.add_argument("--record-fps", type=float, default=15.0,
                    help="playback rate for --record (default 15) -- a fixed rate, "
                         "not the measured inference fps")
    sp.add_argument("--max-frames", type=int, default=0)
    sp.add_argument("--no-thread", action="store_true",
                    help="serial loop (default: infer on a worker thread)")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("bench", parents=[common], help="per-frame latency / fps")
    sp.add_argument("--backend", default=None, choices=["mlx", "torch", "torch-mps", "dummy"],
                    help="default: $SEGFORMER_BACKEND, else mlx")
    sp.add_argument("--model", default=None, help="registry name from models.toml")
    sp.add_argument("--checkpoint", default=None, help="override -- see `run --help`")
    sp.add_argument("--frames", type=int, default=300)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("replay", parents=[common],
                        help="publish a recorded session on ZMQ (no camera needed)")
    sp.add_argument("session", help="session folder (meta.json + chunk_*.npz)")
    sp.add_argument("--fast", action="store_true", help="ignore recorded timing, go flat out")
    sp.add_argument("--loop", action="store_true", help="restart at the end")
    sp.add_argument("--compress", action="store_true", help="lz4-compress arrays on the wire")
    sp.add_argument("--info", action="store_true", help="print recording stats and exit")
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("models", help="manage model weights (registry)")
    msub = sp.add_subparsers(dest="models_cmd", required=True)
    for c in ("list", "where"):
        msub.add_parser(c)
    for c in ("pull", "convert", "verify", "path"):
        mp = msub.add_parser(c)
        mp.add_argument("name", nargs="?", default=None, help="model name (default from models.toml)")
    sp.set_defaults(func=cmd_models)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

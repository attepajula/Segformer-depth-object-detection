"""``segformer`` CLI.

    segformer probe  [--endpoint ...]                 decode one frame, print stats
    segformer run    [--backend mlx] [--overlay|--save DIR] [--max-frames N]
    segformer bench  [--backend mlx] [--frames N]     per-frame latency / fps
    segformer models list|pull|convert|verify|path|where
"""

from __future__ import annotations

import argparse
import sys


def cmd_probe(args) -> int:
    import zmq

    from segformer.wire import decode_frame

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
    from segformer.consumer import FrameConsumer, RateMeter
    from segformer import overlay

    seg = make_segmenter(args.backend)
    seg.load(args.checkpoint)

    has_display = args.overlay or args.save
    if has_display and not args.no_thread:
        from segformer.runner import run_threaded

        print(f"segformer run (threaded): backend={seg.name} endpoint={args.endpoint} "
              f"overlay={args.overlay} save={args.save or '-'}")
        run_threaded(args.endpoint, seg, show=args.overlay, save_dir=args.save,
                     max_frames=args.max_frames)
        return 0

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
            prev_id = frame.frame_id

            result, state = seg.infer(frame, state)
            fps = meter.tick()

            if args.overlay or save_dir is not None:
                img = overlay.render(frame, result, fps=fps)
                if save_dir is not None:
                    _write_png(save_dir / f"{frame.frame_id:06d}.png", img)
                if args.overlay and not overlay.show(img):
                    break
            else:
                print(f"  frame {frame.frame_id}  inst={len(result.instances)}  {fps:4.1f} fps")

            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        seg.close()
    print(f"processed {n} frames")
    return 0


def _write_png(path, rgb) -> None:
    import cv2

    cv2.imwrite(str(path), rgb[:, :, ::-1])


def cmd_bench(args) -> int:
    import time

    from segformer.backends import make_segmenter
    from segformer.consumer import FrameConsumer

    seg = make_segmenter(args.backend)
    t = time.time()
    seg.load()
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
            import mlx.core as mx

            from segformer.model import load_model

            load_model(spec.name)
            print(f"  artifact loads OK ({len(mx.load(str(art)))} tensors)")
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
    sp.add_argument("--backend", default="mlx", choices=["mlx", "torch-mps", "dummy"])
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--overlay", action="store_true", help="OpenCV window")
    sp.add_argument("--save", default=None, metavar="DIR", help="write overlay PNGs")
    sp.add_argument("--max-frames", type=int, default=0)
    sp.add_argument("--no-thread", action="store_true",
                    help="serial loop (default: infer on a worker thread)")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("bench", parents=[common], help="per-frame latency / fps")
    sp.add_argument("--backend", default="mlx", choices=["mlx", "torch-mps", "dummy"])
    sp.add_argument("--frames", type=int, default=300)
    sp.set_defaults(func=cmd_bench)

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

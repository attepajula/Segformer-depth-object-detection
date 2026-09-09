"""``seqformer`` CLI — SCAFFOLD.

Planned subcommands:
    seqformer run   --endpoint tcp://127.0.0.1:5556 --backend torch-mps [--overlay]
    seqformer bench --backend torch-mps --frames 300
    seqformer probe --endpoint ...        # subscribe, decode, print frame stats

Only ``probe`` is wired (it needs no model); the rest raise NotImplementedError.
"""

from __future__ import annotations

import argparse
import sys


def cmd_probe(args) -> int:
    import zmq

    from seqformer.wire import decode_frame

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
    from seqformer.backends import make_segmenter
    from seqformer.consumer import FrameConsumer, RateMeter
    from seqformer import overlay

    seg = make_segmenter(args.backend)
    seg.load(args.checkpoint)
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

    print(f"seqformer run: backend={seg.name} endpoint={args.endpoint} "
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
    raise NotImplementedError("seqformer bench — not wired yet")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--endpoint", default="tcp://127.0.0.1:5556")

    p = argparse.ArgumentParser(prog="seqformer", description=__doc__, parents=[common],
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
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("bench", parents=[common], help="[stub] fps benchmark")
    sp.add_argument("--backend", default="mlx", choices=["mlx", "torch-mps", "dummy"])
    sp.add_argument("--frames", type=int, default=300)
    sp.set_defaults(func=cmd_bench)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

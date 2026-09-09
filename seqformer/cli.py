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
    raise NotImplementedError("seqformer run — consumer loop not wired yet")


def cmd_bench(args) -> int:
    raise NotImplementedError("seqformer bench — not wired yet")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="seqformer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", help="subscribe, decode one frame, print stats")
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("run", help="[stub] consume + segment + overlay")
    sp.add_argument("--backend", default="torch-mps", choices=["torch-mps", "mlx", "dummy"])
    sp.add_argument("--overlay", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("bench", help="[stub] fps benchmark")
    sp.add_argument("--backend", default="torch-mps", choices=["torch-mps", "mlx", "dummy"])
    sp.add_argument("--frames", type=int, default=300)
    sp.set_defaults(func=cmd_bench)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

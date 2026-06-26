from __future__ import annotations

import argparse
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="npu-ctrl", description="Clean-room Strix Halo NPU control plane")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("discover", help="discover NPU devices and write devices.json")
    sub.add_parser("status", help="print summarized device/toolchain status")
    toolchain = sub.add_parser("toolchain", help="toolchain commands")
    toolchain_sub = toolchain.add_subparsers(dest="toolchain_command")
    toolchain_sub.add_parser("probe", help="probe Peano, IRON, and Chess availability")
    kernels = sub.add_parser("kernels", help="kernel registry commands")
    kernels_sub = kernels.add_subparsers(dest="kernels_command")
    kernels_sub.add_parser("list", help="list registered kernels")
    bench = sub.add_parser("bench", help="benchmark commands")
    bench_sub = bench.add_subparsers(dest="bench_command")
    bench_sub.add_parser("list", help="list benchmark runs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if exc.code is not None else 0
    if args.command is None:
        parser.print_help()
        return 0
    parser.error(f"command not implemented yet: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

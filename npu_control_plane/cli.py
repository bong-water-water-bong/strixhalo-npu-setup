from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .benchmark import BenchmarkRecorder
from .discovery import discover_devices
from .metadata import MetadataStore
from .registry import KernelRegistry
from .toolchains import probe_toolchains


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
    register = kernels_sub.add_parser("register", help="register a kernel artifact")
    register.add_argument("--name", required=True)
    register.add_argument("--artifact", required=True)
    register.add_argument("--dtype", required=True)
    register.add_argument("--shape", required=True)
    register.add_argument("--toolchain", required=True)
    register.add_argument("--source-hash")
    bench = sub.add_parser("bench", help="benchmark commands")
    bench_sub = bench.add_subparsers(dest="bench_command")
    bench_sub.add_parser("list", help="list benchmark runs")
    bench_run = bench_sub.add_parser("run", help="record timings for a command")
    bench_run.add_argument("--label", required=True)
    bench_run.add_argument("--warmup", type=int, default=1)
    bench_run.add_argument("--iters", type=int, default=5)
    bench_run.add_argument("cmd", nargs=argparse.REMAINDER)
    return parser


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if exc.code is not None else 0
    store = MetadataStore()
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "discover":
        _print_json(discover_devices(store))
        return 0
    if args.command == "status":
        devices = store.read_json("devices.json", default={"devices": [], "warnings": ["run npu-ctrl discover"]})
        toolchains = store.read_json("toolchains.json", default={"toolchains": [], "warnings": ["run npu-ctrl toolchain probe"]})
        _print_json({"devices": devices, "toolchains": toolchains})
        return 0
    if args.command == "toolchain" and args.toolchain_command == "probe":
        _print_json(probe_toolchains(store))
        return 0
    if args.command == "kernels":
        registry = KernelRegistry(store)
        if args.kernels_command == "list":
            _print_json({"kernels": registry.list()})
            return 0
        if args.kernels_command == "register":
            _print_json(
                registry.register(
                    name=args.name,
                    artifact=Path(args.artifact),
                    dtype=args.dtype,
                    shape=args.shape,
                    toolchain=args.toolchain,
                    source_hash=args.source_hash,
                )
            )
            return 0
    if args.command == "bench":
        recorder = BenchmarkRecorder(store)
        if args.bench_command == "list":
            _print_json({"runs": recorder.list_runs()})
            return 0
        if args.bench_command == "run":
            command = list(args.cmd)
            if command and command[0] == "--":
                command = command[1:]
            if not command:
                parser.error("bench run requires a command after --")
            _print_json(recorder.record_command(args.label, command, args.warmup, args.iters))
            return 0
    parser.error(f"command not implemented yet: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

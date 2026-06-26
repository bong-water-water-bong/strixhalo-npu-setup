from __future__ import annotations

import os
from typing import Any

from .metadata import MetadataStore
from .probe import run_command, which


def _peano_report() -> dict[str, Any]:
    clang = which("clang++")
    peano_dir = os.environ.get("PEANO_INSTALL_DIR")
    if not clang:
        return {"name": "peano", "available": False, "license_required": False, "reason": "clang++ not found"}
    version = run_command([clang, "--version"])
    return {
        "name": "peano",
        "available": True,
        "license_required": False,
        "path": peano_dir or clang,
        "clang": clang,
        "version": (version.stdout or version.stderr).splitlines()[0] if (version.stdout or version.stderr) else "unknown",
    }


def _iron_report() -> dict[str, Any]:
    python = which("python3")
    if not python:
        return {"name": "iron", "available": False, "license_required": False, "reason": "python3 not found"}
    result = run_command([python, "-c", "import aie.iron; print('iron-ok')"])
    if result.returncode != 0:
        return {
            "name": "iron",
            "available": False,
            "license_required": False,
            "python": python,
            "reason": "python3 cannot import aie.iron",
        }
    return {"name": "iron", "available": True, "license_required": False, "python": python}


def _chess_report() -> dict[str, Any]:
    chess = which("xchesscc")
    license_file = os.environ.get("XILINXD_LICENSE_FILE")
    base = {
        "name": "chess",
        "license_required": True,
        "public_clean_room_use": "benchmark reference only; do not redistribute artifacts",
    }
    if not chess:
        return {**base, "available": False, "reason": "xchesscc not found"}
    if not license_file:
        return {**base, "available": False, "path": chess, "reason": "XILINXD_LICENSE_FILE not set"}
    version = run_command([chess, "--version"])
    return {
        **base,
        "available": version.returncode == 0,
        "path": chess,
        "license_file_configured": True,
        "version": (version.stdout or version.stderr).splitlines()[0] if (version.stdout or version.stderr) else "unknown",
        "reason": "" if version.returncode == 0 else "xchesscc --version failed",
    }


def probe_toolchains(store: MetadataStore | None = None) -> dict[str, Any]:
    store = store or MetadataStore()
    report = {"toolchains": [_peano_report(), _iron_report(), _chess_report()]}
    store.write_json("toolchains.json", data=report)
    return report

from __future__ import annotations

import re
from typing import Any

from .metadata import MetadataStore
from .probe import run_command

_DEVICE_RE = re.compile(r"\[(?P<bdf>[0-9a-fA-F:.]+)\].*\|(?P<name>RyzenAI-[^| ]+)")


def _driver_loaded() -> bool:
    result = run_command(["lsmod"])
    return result.returncode == 0 and "amdxdna" in result.stdout


def discover_devices(store: MetadataStore | None = None) -> dict[str, Any]:
    store = store or MetadataStore()
    warnings: list[str] = []
    devices: list[dict[str, Any]] = []
    driver_loaded = _driver_loaded()
    result = run_command(["xrt-smi", "examine"])
    if result.returncode != 0:
        warnings.append(result.stderr.strip() or "xrt-smi examine failed")
    else:
        for line in result.stdout.splitlines():
            match = _DEVICE_RE.search(line)
            if not match:
                continue
            devices.append(
                {
                    "id": len(devices),
                    "bdf": match.group("bdf"),
                    "name": match.group("name").strip(),
                    "driver": "amdxdna",
                    "runtime": "xrt",
                    "tile_count": 32,
                    "peak_tflops_claimed": 31.2,
                }
            )
    report = {"devices": devices, "driver_loaded": driver_loaded, "warnings": warnings}
    store.write_json("devices.json", data=report)
    return report

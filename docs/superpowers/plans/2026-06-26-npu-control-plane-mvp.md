# NPU Control Plane MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public clean-room `npu-ctrl` MVP that discovers the Strix Halo NPU/toolchains, stores metadata, registers kernel artifacts, and records repeatable benchmark runs.

**Architecture:** Implement a small Python package with focused modules for metadata storage, device discovery, toolchain probing, kernel registry, benchmark recording, and CLI routing. Use only public OS/runtime signals and user-provided paths; do not inspect or copy proprietary Chess internals. Persist state under `~/.npu_control_plane/store` by default, with `NPU_CTRL_STORE` override for tests and reproducibility.

**Tech Stack:** Python 3.10+, standard library, `pytest` for tests, JSON metadata files, subprocess-based probing of public commands (`xrt-smi`, `lsmod`, `python`, `clang++`).

## Global Constraints

- Keep the project public/open-source clean-room: no proprietary Chess code, headers, binaries, generated artifacts, decompilation output, private license files, or unpublished internal documentation.
- Allowed inputs are public AMD/Xilinx docs, public repositories, public examples, public benchmark methodology, and independently authored experiments.
- Black-box Chess numbers may be recorded only as high-level benchmark references when legally obtained; do not redistribute Chess-produced artifacts.
- Start narrow: Phase 1 is measurement harness and `npu-ctrl` MVP, not a full compiler.
- Use JSON metadata under `~/.npu_control_plane/store` by default.
- Provide deterministic tests that do not require an actual NPU, XRT install, Peano install, IRON install, or Chess license.
- CLI commands must never fail just because optional toolchains are missing; they should report `available: false` with a reason.

---

## File Structure

Create these files:

- `pyproject.toml` — package metadata, pytest config, `npu-ctrl` console script.
- `npu_control_plane/__init__.py` — package version and public module marker.
- `npu_control_plane/metadata.py` — JSON store path resolution, atomic-ish reads/writes, directory creation.
- `npu_control_plane/probe.py` — tiny subprocess/shutil helpers used by discovery and toolchain probing.
- `npu_control_plane/discovery.py` — NPU/driver/XRT discovery from public commands.
- `npu_control_plane/toolchains.py` — Peano, IRON, and Chess availability reports.
- `npu_control_plane/registry.py` — kernel artifact registry keyed by name/shape/dtype/toolchain/source hash.
- `npu_control_plane/benchmark.py` — benchmark run schema and command timing helper.
- `npu_control_plane/cli.py` — `npu-ctrl` command surface.
- `tests/test_metadata.py` — metadata store tests.
- `tests/test_discovery_toolchains.py` — mocked discovery/toolchain tests.
- `tests/test_registry.py` — kernel registry tests.
- `tests/test_benchmark_cli.py` — benchmark and CLI tests.
- `README.md` — add a short `npu-ctrl` MVP usage section.

Do not create compiler backend code in this phase. The MVP is infrastructure for later compiler experiments.

---

### Task 1: Project Scaffold and Metadata Store

**Files:**
- Create: `pyproject.toml`
- Create: `npu_control_plane/__init__.py`
- Create: `npu_control_plane/metadata.py`
- Create: `npu_control_plane/cli.py`
- Create: `tests/test_metadata.py`
- Create: `tests/test_benchmark_cli.py` initially with CLI smoke tests only

**Interfaces:**
- Produces: `npu_control_plane.metadata.default_store_dir() -> pathlib.Path`
- Produces: `npu_control_plane.metadata.MetadataStore(root: Path | None = None)`
- Produces: `MetadataStore.path(*parts: str) -> Path`
- Produces: `MetadataStore.read_json(*parts: str, default: Any = None) -> Any`
- Produces: `MetadataStore.write_json(*parts: str, data: Any) -> Path`
- Produces: `npu_control_plane.cli.main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write failing metadata tests**

Create `tests/test_metadata.py`:

```python
import json
from pathlib import Path

from npu_control_plane.metadata import MetadataStore, default_store_dir


def test_default_store_dir_uses_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "custom-store"))

    assert default_store_dir() == tmp_path / "custom-store"


def test_metadata_store_writes_pretty_json(tmp_path):
    store = MetadataStore(tmp_path / "store")

    written = store.write_json("devices.json", data={"devices": [{"name": "RyzenAI-npu5"}]})

    assert written == tmp_path / "store" / "devices.json"
    assert json.loads(written.read_text()) == {"devices": [{"name": "RyzenAI-npu5"}]}
    assert written.read_text().endswith("\n")
    assert "  \"devices\"" in written.read_text()


def test_metadata_store_read_json_returns_default_for_missing_file(tmp_path):
    store = MetadataStore(tmp_path / "store")

    assert store.read_json("missing.json", default={"ok": False}) == {"ok": False}


def test_metadata_store_rejects_paths_that_escape_root(tmp_path):
    store = MetadataStore(tmp_path / "store")

    try:
        store.path("..", "outside.json")
    except ValueError as exc:
        assert "escape" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Write failing CLI smoke tests**

Create `tests/test_benchmark_cli.py` with only this initial test:

```python
from npu_control_plane.cli import main


def test_cli_help_returns_zero(capsys):
    code = main(["--help"])

    captured = capsys.readouterr()
    assert code == 0
    assert "npu-ctrl" in captured.out
    assert "discover" in captured.out
```

- [ ] **Step 3: Run tests and verify they fail because package does not exist**

Run:

```bash
cd /home/bcloud/strixhalo-npu-setup
python3 -m pytest tests/test_metadata.py tests/test_benchmark_cli.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'npu_control_plane'`.

- [ ] **Step 4: Add package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "strixhalo-npu-control-plane"
version = "0.1.0"
description = "Clean-room control plane MVP for Strix Halo NPU experiments"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [{ name = "Strix Halo NPU Contributors" }]
dependencies = []

[project.scripts]
npu-ctrl = "npu_control_plane.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 5: Add package marker**

Create `npu_control_plane/__init__.py`:

```python
"""Clean-room Strix Halo NPU control plane MVP."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Implement metadata store**

Create `npu_control_plane/metadata.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ENV_STORE = "NPU_CTRL_STORE"


def default_store_dir() -> Path:
    """Return the metadata store directory.

    Tests and reproducible runs can override the default with NPU_CTRL_STORE.
    """

    override = os.environ.get(ENV_STORE)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".npu_control_plane" / "store"


class MetadataStore:
    """Small JSON metadata store rooted at one directory."""

    def __init__(self, root: Path | None = None):
        self.root = (root or default_store_dir()).expanduser().resolve()

    def path(self, *parts: str) -> Path:
        candidate = self.root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"metadata path would escape store root: {candidate}") from exc
        return candidate

    def read_json(self, *parts: str, default: Any = None) -> Any:
        path = self.path(*parts)
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_json(self, *parts: str, data: Any) -> Path:
        path = self.path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
        return path
```

- [ ] **Step 7: Implement CLI skeleton**

Create `npu_control_plane/cli.py`:

```python
from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    parser.error(f"command not implemented yet: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run tests and verify scaffold passes**

Run:

```bash
cd /home/bcloud/strixhalo-npu-setup
python3 -m pytest tests/test_metadata.py tests/test_benchmark_cli.py -q
```

Expected: `5 passed`.

- [ ] **Step 9: Commit scaffold**

Run:

```bash
git add pyproject.toml npu_control_plane tests/test_metadata.py tests/test_benchmark_cli.py
git commit -m "feat: add npu control plane scaffold"
```

Expected: commit succeeds.

---

### Task 2: Device Discovery and Toolchain Probing

**Files:**
- Create: `npu_control_plane/probe.py`
- Create: `npu_control_plane/discovery.py`
- Create: `npu_control_plane/toolchains.py`
- Modify: `npu_control_plane/cli.py`
- Create: `tests/test_discovery_toolchains.py`

**Interfaces:**
- Consumes: `MetadataStore.write_json()` from Task 1
- Produces: `probe.CommandResult(args: list[str], returncode: int, stdout: str, stderr: str)`
- Produces: `probe.run_command(args: Sequence[str], timeout: int = 5) -> CommandResult`
- Produces: `probe.which(name: str) -> str | None`
- Produces: `discovery.discover_devices(store: MetadataStore | None = None) -> dict[str, Any]`
- Produces: `toolchains.probe_toolchains(store: MetadataStore | None = None) -> dict[str, Any]`

- [ ] **Step 1: Write failing discovery/toolchain tests**

Create `tests/test_discovery_toolchains.py`:

```python
from npu_control_plane.discovery import discover_devices
from npu_control_plane.metadata import MetadataStore
from npu_control_plane.toolchains import probe_toolchains


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.args = []


def test_discover_devices_parses_ryzenai_from_xrt_smi(monkeypatch, tmp_path):
    def fake_run(args, timeout=5):
        if args[:2] == ["xrt-smi", "examine"]:
            return FakeResult(stdout="[0000:c6:00.1]  |RyzenAI-npu5  |\n")
        if args[:1] == ["lsmod"]:
            return FakeResult(stdout="amdxdna 69632 0\n")
        return FakeResult(returncode=1, stderr="unexpected command")

    monkeypatch.setattr("npu_control_plane.discovery.run_command", fake_run)
    store = MetadataStore(tmp_path / "store")

    report = discover_devices(store)

    assert report["devices"] == [
        {
            "id": 0,
            "bdf": "0000:c6:00.1",
            "name": "RyzenAI-npu5",
            "driver": "amdxdna",
            "runtime": "xrt",
            "tile_count": 32,
            "peak_tflops_claimed": 31.2,
        }
    ]
    assert store.read_json("devices.json")["devices"][0]["name"] == "RyzenAI-npu5"


def test_discover_devices_reports_missing_xrt_without_failure(monkeypatch, tmp_path):
    def fake_run(args, timeout=5):
        if args[:2] == ["xrt-smi", "examine"]:
            return FakeResult(returncode=127, stderr="xrt-smi: not found")
        if args[:1] == ["lsmod"]:
            return FakeResult(stdout="")
        return FakeResult(returncode=1)

    monkeypatch.setattr("npu_control_plane.discovery.run_command", fake_run)
    store = MetadataStore(tmp_path / "store")

    report = discover_devices(store)

    assert report["devices"] == []
    assert report["driver_loaded"] is False
    assert "xrt-smi" in report["warnings"][0]


def test_probe_toolchains_detects_peano_iron_and_chess(monkeypatch, tmp_path):
    monkeypatch.setenv("PEANO_INSTALL_DIR", "/opt/peano")
    monkeypatch.setenv("XILINXD_LICENSE_FILE", "/licenses/Xilinx.lic")

    def fake_which(name):
        return {
            "clang++": "/opt/peano/bin/clang++",
            "xchesscc": "/opt/chess/bin/xchesscc",
            "python3": "/usr/bin/python3",
        }.get(name)

    def fake_run(args, timeout=5):
        if args[0] == "/usr/bin/python3":
            return FakeResult(stdout="iron-ok\n")
        if args[0] == "/opt/peano/bin/clang++":
            return FakeResult(stdout="clang version 21.0.0\n")
        if args[0] == "/opt/chess/bin/xchesscc":
            return FakeResult(stdout="Chess Compiler\n")
        return FakeResult(returncode=1, stderr="unexpected")

    monkeypatch.setattr("npu_control_plane.toolchains.which", fake_which)
    monkeypatch.setattr("npu_control_plane.toolchains.run_command", fake_run)
    store = MetadataStore(tmp_path / "store")

    report = probe_toolchains(store)

    by_name = {item["name"]: item for item in report["toolchains"]}
    assert by_name["peano"]["available"] is True
    assert by_name["iron"]["available"] is True
    assert by_name["chess"]["available"] is True
    assert by_name["chess"]["public_clean_room_use"] == "benchmark reference only; do not redistribute artifacts"
    assert store.read_json("toolchains.json") == report


def test_probe_toolchains_reports_missing_optional_tools(monkeypatch, tmp_path):
    monkeypatch.delenv("PEANO_INSTALL_DIR", raising=False)
    monkeypatch.delenv("XILINXD_LICENSE_FILE", raising=False)
    monkeypatch.setattr("npu_control_plane.toolchains.which", lambda name: None)
    store = MetadataStore(tmp_path / "store")

    report = probe_toolchains(store)

    by_name = {item["name"]: item for item in report["toolchains"]}
    assert by_name["peano"]["available"] is False
    assert "clang++" in by_name["peano"]["reason"]
    assert by_name["iron"]["available"] is False
    assert by_name["chess"]["available"] is False
    assert by_name["chess"]["license_required"] is True
```

- [ ] **Step 2: Run tests and verify they fail because modules do not exist**

Run:

```bash
python3 -m pytest tests/test_discovery_toolchains.py -q
```

Expected: FAIL with `ModuleNotFoundError` or import errors for `discovery` / `toolchains`.

- [ ] **Step 3: Add subprocess helpers**

Create `npu_control_plane/probe.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def which(name: str) -> str | None:
    return shutil.which(name)


def run_command(args: Sequence[str], timeout: int = 5) -> CommandResult:
    try:
        completed = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(list(args), completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        return CommandResult(list(args), 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(list(args), 124, exc.stdout or "", exc.stderr or f"timed out after {timeout}s")
```

- [ ] **Step 4: Implement device discovery**

Create `npu_control_plane/discovery.py`:

```python
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
```

- [ ] **Step 5: Implement toolchain probing**

Create `npu_control_plane/toolchains.py`:

```python
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
```

- [ ] **Step 6: Wire CLI discover/status/toolchain probe**

Replace `npu_control_plane/cli.py` with:

```python
from __future__ import annotations

import argparse
import json
from typing import Sequence

from .discovery import discover_devices
from .metadata import MetadataStore
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
    bench = sub.add_parser("bench", help="benchmark commands")
    bench_sub = bench.add_subparsers(dest="bench_command")
    bench_sub.add_parser("list", help="list benchmark runs")
    return parser


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    parser.error(f"command not implemented yet: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_discovery_toolchains.py tests/test_benchmark_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Run real host probes without requiring success**

Run:

```bash
NPU_CTRL_STORE=/tmp/npu-ctrl-plan-check python3 -m npu_control_plane.cli discover
NPU_CTRL_STORE=/tmp/npu-ctrl-plan-check python3 -m npu_control_plane.cli toolchain probe
NPU_CTRL_STORE=/tmp/npu-ctrl-plan-check python3 -m npu_control_plane.cli status
```

Expected: each command exits `0` and prints JSON. Missing optional tools appear as `available: false` or warnings, not Python tracebacks.

- [ ] **Step 9: Commit discovery and toolchain probing**

Run:

```bash
git add npu_control_plane tests/test_discovery_toolchains.py tests/test_benchmark_cli.py
git commit -m "feat: add npu discovery and toolchain probing"
```

Expected: commit succeeds.

---

### Task 3: Kernel Artifact Registry

**Files:**
- Create: `npu_control_plane/registry.py`
- Modify: `npu_control_plane/cli.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- Consumes: `MetadataStore`
- Produces: `registry.KernelRegistry(store: MetadataStore | None = None)`
- Produces: `KernelRegistry.list() -> list[dict[str, Any]]`
- Produces: `KernelRegistry.register(name: str, artifact: Path, dtype: str, shape: str, toolchain: str, source_hash: str | None = None) -> dict[str, Any]`
- Produces CLI: `npu-ctrl kernels list`
- Produces CLI: `npu-ctrl kernels register --name NAME --artifact PATH --dtype DTYPE --shape SHAPE --toolchain TOOLCHAIN [--source-hash HASH]`

- [ ] **Step 1: Write failing registry tests**

Create `tests/test_registry.py`:

```python
import json

from npu_control_plane.metadata import MetadataStore
from npu_control_plane.registry import KernelRegistry


def test_registry_starts_empty(tmp_path):
    registry = KernelRegistry(MetadataStore(tmp_path / "store"))

    assert registry.list() == []


def test_registry_registers_artifact_with_content_hash(tmp_path):
    artifact = tmp_path / "kernel.elf"
    artifact.write_bytes(b"public-test-artifact")
    registry = KernelRegistry(MetadataStore(tmp_path / "store"))

    record = registry.register(
        name="vec_add",
        artifact=artifact,
        dtype="i32",
        shape="N=64",
        toolchain="peano",
    )

    assert record["key"].startswith("vec_add__N=64__i32__peano__")
    assert record["name"] == "vec_add"
    assert record["artifact"].startswith("registry/kernels/artifacts/")
    copied = tmp_path / "store" / record["artifact"]
    assert copied.read_bytes() == b"public-test-artifact"
    index = json.loads((tmp_path / "store" / "registry" / "kernels" / "index.json").read_text())
    assert index["kernels"] == [record]


def test_registry_replaces_same_key_without_duplicates(tmp_path):
    artifact = tmp_path / "kernel.elf"
    artifact.write_bytes(b"public-test-artifact")
    registry = KernelRegistry(MetadataStore(tmp_path / "store"))

    first = registry.register("vec_add", artifact, "i32", "N=64", "peano", source_hash="abc")
    second = registry.register("vec_add", artifact, "i32", "N=64", "peano", source_hash="abc")

    assert first["key"] == second["key"]
    assert registry.list() == [second]
```

- [ ] **Step 2: Run tests and verify they fail because registry is missing**

Run:

```bash
python3 -m pytest tests/test_registry.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'npu_control_plane.registry'`.

- [ ] **Step 3: Implement registry**

Create `npu_control_plane/registry.py`:

```python
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from .metadata import MetadataStore


class KernelRegistry:
    def __init__(self, store: MetadataStore | None = None):
        self.store = store or MetadataStore()

    def _index(self) -> dict[str, Any]:
        return self.store.read_json("registry", "kernels", "index.json", default={"kernels": []})

    def _write_index(self, index: dict[str, Any]) -> None:
        self.store.write_json("registry", "kernels", "index.json", data=index)

    def list(self) -> list[dict[str, Any]]:
        return list(self._index().get("kernels", []))

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def register(
        self,
        name: str,
        artifact: Path,
        dtype: str,
        shape: str,
        toolchain: str,
        source_hash: str | None = None,
    ) -> dict[str, Any]:
        artifact = artifact.expanduser().resolve()
        if not artifact.is_file():
            raise FileNotFoundError(str(artifact))
        artifact_hash = self._hash_file(artifact)
        source = source_hash or artifact_hash
        key = f"{name}__{shape}__{dtype}__{toolchain}__{source[:12]}"
        suffix = artifact.suffix or ".bin"
        relative_artifact = f"registry/kernels/artifacts/{key}{suffix}"
        destination = self.store.path(*relative_artifact.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, destination)
        record = {
            "key": key,
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "toolchain": toolchain,
            "source_hash": source,
            "artifact_hash": artifact_hash,
            "artifact": relative_artifact,
        }
        index = self._index()
        kernels = [item for item in index.get("kernels", []) if item.get("key") != key]
        kernels.append(record)
        index["kernels"] = sorted(kernels, key=lambda item: item["key"])
        self._write_index(index)
        return record
```

- [ ] **Step 4: Wire registry CLI**

Edit `npu_control_plane/cli.py`:

1. Add imports:

```python
from pathlib import Path
from .registry import KernelRegistry
```

2. Replace the `kernels` parser block with:

```python
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
```

3. Add this branch before the final `parser.error(...)`:

```python
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
```

- [ ] **Step 5: Add CLI registry test**

Append to `tests/test_registry.py`:

```python
from npu_control_plane.cli import main


def test_cli_kernels_list_uses_store_env(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

    code = main(["kernels", "list"])

    captured = capsys.readouterr()
    assert code == 0
    assert '"kernels": []' in captured.out
```

- [ ] **Step 6: Run registry tests**

Run:

```bash
python3 -m pytest tests/test_registry.py -q
```

Expected: all registry tests pass.

- [ ] **Step 7: Run all tests so far**

Run:

```bash
python3 -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit registry**

Run:

```bash
git add npu_control_plane tests/test_registry.py
git commit -m "feat: add kernel artifact registry"
```

Expected: commit succeeds.

---

### Task 4: Benchmark Run Recording

**Files:**
- Create: `npu_control_plane/benchmark.py`
- Modify: `npu_control_plane/cli.py`
- Modify: `tests/test_benchmark_cli.py`

**Interfaces:**
- Consumes: `MetadataStore`
- Produces: `benchmark.BenchmarkRecorder(store: MetadataStore | None = None)`
- Produces: `BenchmarkRecorder.list_runs() -> list[dict[str, Any]]`
- Produces: `BenchmarkRecorder.record_command(label: str, command: list[str], warmup: int, iters: int) -> dict[str, Any]`
- Produces CLI: `npu-ctrl bench list`
- Produces CLI: `npu-ctrl bench run --label LABEL -- COMMAND...`

- [ ] **Step 1: Write failing benchmark tests**

Replace `tests/test_benchmark_cli.py` with:

```python
from npu_control_plane.benchmark import BenchmarkRecorder
from npu_control_plane.cli import main
from npu_control_plane.metadata import MetadataStore


def test_cli_help_returns_zero(capsys):
    code = main(["--help"])

    captured = capsys.readouterr()
    assert code == 0
    assert "npu-ctrl" in captured.out
    assert "discover" in captured.out


def test_benchmark_recorder_records_command(monkeypatch, tmp_path):
    times = iter([10.0, 10.1, 20.0, 20.2, 30.0, 30.4])

    def fake_perf_counter():
        return next(times)

    calls = []

    def fake_run_command(command, timeout=300):
        calls.append(list(command))
        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return Result()

    monkeypatch.setattr("npu_control_plane.benchmark.perf_counter", fake_perf_counter)
    monkeypatch.setattr("npu_control_plane.benchmark.run_command", fake_run_command)
    recorder = BenchmarkRecorder(MetadataStore(tmp_path / "store"))

    record = recorder.record_command(label="echo-ok", command=["echo", "ok"], warmup=1, iters=2)

    assert calls == [["echo", "ok"], ["echo", "ok"], ["echo", "ok"]]
    assert record["label"] == "echo-ok"
    assert record["warmup"] == 1
    assert record["iters"] == 2
    assert record["durations_ms"] == [199.9999999999993, 399.9999999999986]
    assert record["returncode"] == 0
    assert recorder.list_runs() == [record]


def test_cli_bench_list_uses_store_env(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

    code = main(["bench", "list"])

    captured = capsys.readouterr()
    assert code == 0
    assert '"runs": []' in captured.out
```

- [ ] **Step 2: Run tests and verify benchmark module is missing**

Run:

```bash
python3 -m pytest tests/test_benchmark_cli.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'npu_control_plane.benchmark'`.

- [ ] **Step 3: Implement benchmark recorder**

Create `npu_control_plane/benchmark.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from statistics import median
from time import perf_counter
from typing import Any

from .metadata import MetadataStore
from .probe import run_command


class BenchmarkRecorder:
    def __init__(self, store: MetadataStore | None = None):
        self.store = store or MetadataStore()

    def list_runs(self) -> list[dict[str, Any]]:
        summary = self.store.read_json("benchmarks", "summary.json", default={"runs": []})
        return list(summary.get("runs", []))

    def _write_summary(self, runs: list[dict[str, Any]]) -> None:
        self.store.write_json("benchmarks", "summary.json", data={"runs": runs})

    def record_command(self, label: str, command: list[str], warmup: int, iters: int) -> dict[str, Any]:
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        if iters <= 0:
            raise ValueError("iters must be > 0")
        durations_ms: list[float] = []
        last_result = None
        for index in range(warmup + iters):
            start = perf_counter()
            last_result = run_command(command, timeout=300)
            elapsed_ms = (perf_counter() - start) * 1000.0
            if index >= warmup:
                durations_ms.append(elapsed_ms)
            if last_result.returncode != 0:
                break
        run_id = str(uuid.uuid4())
        record = {
            "run_id": run_id,
            "label": label,
            "command": command,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warmup": warmup,
            "iters": iters,
            "durations_ms": durations_ms,
            "median_ms": median(durations_ms) if durations_ms else None,
            "returncode": last_result.returncode if last_result else None,
            "stdout_tail": (last_result.stdout[-4000:] if last_result else ""),
            "stderr_tail": (last_result.stderr[-4000:] if last_result else ""),
        }
        self.store.write_json("benchmarks", "runs", f"{run_id}.json", data=record)
        runs = [item for item in self.list_runs() if item.get("run_id") != run_id]
        runs.append(record)
        self._write_summary(runs)
        return record
```

- [ ] **Step 4: Wire benchmark CLI**

Edit `npu_control_plane/cli.py`:

1. Add import:

```python
from .benchmark import BenchmarkRecorder
```

2. Replace the `bench` parser block with:

```python
    bench = sub.add_parser("bench", help="benchmark commands")
    bench_sub = bench.add_subparsers(dest="bench_command")
    bench_sub.add_parser("list", help="list benchmark runs")
    bench_run = bench_sub.add_parser("run", help="record timings for a command")
    bench_run.add_argument("--label", required=True)
    bench_run.add_argument("--warmup", type=int, default=1)
    bench_run.add_argument("--iters", type=int, default=5)
    bench_run.add_argument("command", nargs=argparse.REMAINDER)
```

3. Add this branch before the final `parser.error(...)`:

```python
    if args.command == "bench":
        recorder = BenchmarkRecorder(store)
        if args.bench_command == "list":
            _print_json({"runs": recorder.list_runs()})
            return 0
        if args.bench_command == "run":
            command = list(args.command)
            if command and command[0] == "--":
                command = command[1:]
            if not command:
                parser.error("bench run requires a command after --")
            _print_json(recorder.record_command(args.label, command, args.warmup, args.iters))
            return 0
```

- [ ] **Step 5: Run benchmark tests**

Run:

```bash
python3 -m pytest tests/test_benchmark_cli.py -q
```

Expected: all benchmark/CLI tests pass.

- [ ] **Step 6: Run real benchmark smoke test**

Run:

```bash
NPU_CTRL_STORE=/tmp/npu-ctrl-bench-smoke python3 -m npu_control_plane.cli bench run --label python-version --warmup 0 --iters 1 -- python3 --version
NPU_CTRL_STORE=/tmp/npu-ctrl-bench-smoke python3 -m npu_control_plane.cli bench list
```

Expected: first command records one run with `returncode: 0`; second command prints a `runs` array with that record.

- [ ] **Step 7: Run full tests**

Run:

```bash
python3 -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit benchmark recorder**

Run:

```bash
git add npu_control_plane tests/test_benchmark_cli.py
git commit -m "feat: add benchmark run recording"
```

Expected: commit succeeds.

---

### Task 5: README Usage and Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: all CLI commands from Tasks 1-4.
- Produces: documented MVP workflow for public clean-room use.

- [ ] **Step 1: Add README usage section**

Append this section to `README.md`:

```markdown
---

## Unified Control Plane MVP

This repository includes a public clean-room `npu-ctrl` MVP for coordinating Strix Halo NPU experiments without depending on proprietary compiler internals.

### Install for local development

```bash
cd ~/strixhalo-npu-setup
python3 -m pip install -e .
```

### Use an isolated metadata store

```bash
export NPU_CTRL_STORE=/tmp/npu-control-plane-demo
```

If `NPU_CTRL_STORE` is not set, metadata is written to:

```text
~/.npu_control_plane/store
```

### Discover device and toolchain state

```bash
npu-ctrl discover
npu-ctrl toolchain probe
npu-ctrl status
```

Missing optional tools are reported as unavailable instead of crashing. Chess is treated as a benchmark-reference-only toolchain for public clean-room work; do not redistribute Chess artifacts or private license files.

### Register a public kernel artifact

```bash
printf 'public artifact example' > /tmp/example-kernel.elf
npu-ctrl kernels register \
  --name vec_add \
  --artifact /tmp/example-kernel.elf \
  --dtype i32 \
  --shape N=64 \
  --toolchain peano
npu-ctrl kernels list
```

### Record a repeatable command benchmark

```bash
npu-ctrl bench run --label python-version --warmup 0 --iters 1 -- python3 --version
npu-ctrl bench list
```

The MVP stores JSON records for devices, toolchains, kernel artifacts, and benchmark runs. Later phases will add AIE kernel builders, profile collection, and compiler experiment integration.
```

- [ ] **Step 2: Run tests after docs change**

Run:

```bash
python3 -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run CLI smoke commands against a disposable store**

Run:

```bash
rm -rf /tmp/npu-ctrl-final-smoke
NPU_CTRL_STORE=/tmp/npu-ctrl-final-smoke python3 -m npu_control_plane.cli --help
NPU_CTRL_STORE=/tmp/npu-ctrl-final-smoke python3 -m npu_control_plane.cli discover
NPU_CTRL_STORE=/tmp/npu-ctrl-final-smoke python3 -m npu_control_plane.cli toolchain probe
NPU_CTRL_STORE=/tmp/npu-ctrl-final-smoke python3 -m npu_control_plane.cli kernels list
NPU_CTRL_STORE=/tmp/npu-ctrl-final-smoke python3 -m npu_control_plane.cli bench run --label python-version --warmup 0 --iters 1 -- python3 --version
NPU_CTRL_STORE=/tmp/npu-ctrl-final-smoke python3 -m npu_control_plane.cli bench list
```

Expected: each command exits `0`; optional missing tools appear as JSON warnings or `available: false`; no Python traceback appears.

- [ ] **Step 4: Verify metadata files were created**

Run:

```bash
find /tmp/npu-ctrl-final-smoke -type f | sort
```

Expected output includes:

```text
/tmp/npu-ctrl-final-smoke/benchmarks/runs/<run-id>.json
/tmp/npu-ctrl-final-smoke/benchmarks/summary.json
/tmp/npu-ctrl-final-smoke/devices.json
/tmp/npu-ctrl-final-smoke/registry/kernels/index.json
/tmp/npu-ctrl-final-smoke/toolchains.json
```

If `registry/kernels/index.json` is absent because no kernel was registered in the smoke sequence, run:

```bash
printf 'public artifact example' > /tmp/example-kernel.elf
NPU_CTRL_STORE=/tmp/npu-ctrl-final-smoke python3 -m npu_control_plane.cli kernels register --name vec_add --artifact /tmp/example-kernel.elf --dtype i32 --shape N=64 --toolchain peano
find /tmp/npu-ctrl-final-smoke -type f | sort
```

Expected: `registry/kernels/index.json` and an artifact under `registry/kernels/artifacts/` are present.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add README.md
git commit -m "docs: document npu control plane mvp"
```

Expected: commit succeeds.

- [ ] **Step 6: Final repository verification**

Run:

```bash
git status --short
python3 -m pytest -q
NPU_CTRL_STORE=/tmp/npu-ctrl-final-check python3 -m npu_control_plane.cli status
```

Expected:

- `git status --short` prints nothing.
- `pytest` reports all tests passing.
- `status` exits `0` and prints JSON.

---

## Self-Review Checklist

Spec coverage:

- Clean-room boundary: covered by global constraints, toolchain report wording, and README warning.
- Measurement harness: covered by `BenchmarkRecorder` and `npu-ctrl bench`.
- Control plane MVP: covered by discovery, toolchain probe, metadata store, registry, and CLI.
- JSON metadata store: covered by `MetadataStore` and tests.
- Deterministic tests without NPU: covered through monkeypatching and temporary stores.
- Kernel registry keyed by source hash/shape/dtype/toolchain: covered by `KernelRegistry.register()`.

Implementation order:

- Task 1 establishes package and metadata foundation.
- Task 2 adds public host/toolchain probes.
- Task 3 adds artifact registry.
- Task 4 adds benchmark recording.
- Task 5 documents and verifies the workflow.

Review gates:

- Each task has a failing test first.
- Each task has focused verification commands.
- Each task ends with a commit.

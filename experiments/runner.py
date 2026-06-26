from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npu_control_plane.metadata import MetadataStore


@dataclass
class ExperimentConfig:
    label: str
    kernel_name: str
    shape_str: str
    warmup: int = 3
    iters: int = 20
    toolchain: str = "iron+peano"
    tile_ids: list[int] = field(default_factory=lambda: [0, 2])
    dtype: str = "bfp16"
    store: MetadataStore | None = None

    def __post_init__(self):
        self.store = self.store or MetadataStore()

    @property
    def name(self) -> str:
        return f"{self.kernel_name}__{self.shape_str}__{self.dtype}"


class NpuRunner:
    """Orchestrate kernel compilation, NPU execution, and npu-ctrl recording.

    Does NOT preflight-check for NPU hardware — the caller handles that.
    """

    def __init__(self, config: ExperimentConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

    def compile_kernel(self, source: str, tmp_dir: Path | None = None) -> Path:
        """Generate a compile script or invoke IRON JIT. Returns path to .xclbin or ELF."""
        if self.dry_run:
            out = (tmp_dir or Path(tempfile.mkdtemp())) / f"{self.config.name}.xclbin"
            out.write_text("dry-run artifact")
            return out
        # TODO: invoke IRON JIT pipeline
        raise NotImplementedError("Real NPU execution requires hardware.")

    def bench_command(self, artifact: Path) -> list[str]:
        """Build an npu-ctrl bench run command list."""
        return [
            sys.executable, "-m", "npu_control_plane.cli",
            "bench", "run",
            "--label", self.config.label,
            "--warmup", str(self.config.warmup),
            "--iters", str(self.config.iters),
            "--",
            self._exec_command(artifact),
        ]

    def _exec_command(self, artifact: Path) -> str:
        return f"./run_kernel --xclbin {artifact} --npu-device 0"

    def run(self) -> dict[str, Any]:
        """Full pipeline: compile → exec → record. Returns benchmark record."""
        raise NotImplementedError("Requires NPU hardware. Implement for real runs.")

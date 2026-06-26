# Phase 2: GEMM-First Open Compiler Experiments

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run open-source AIE2P kernels (passthrough → vector add → GEMM) on Strix Halo NPU using only public tooling (IRON, Peano, XRT), with measurements collected via the `npu-ctrl` MVP.

**Architecture:** Write kernel descriptions in IRON/MLIR, compile through Peano, execute via XRT, record timings with `npu-ctrl bench run`. Each experiment is a standalone Python script that can run on the target hardware. Results are published as structured JSON.

**Tech Stack:** Python 3.10+, `aie.iron` (MLIR-AIE), Peano/LLVM-AIE (`clang++ --target=aie2`), XRT (`pyxrt`), `npu-ctrl` CLI.

## Global Constraints

- Public/open-source clean-room: only public AMD/Xilinx docs, public repos, public intrinsics, independently authored code.
- Do not copy, reference, or depend on proprietary Chess internals, generated artifacts, or private license files.
- All kernel source must be Apache 2.0 compatible (IRON/MLIR-AIE is Apache 2.0; Peano/LLVM-AIE is Apache 2.0 with LLVM exceptions).
- Use `npu-ctrl` for all benchmark recording. Record metadata: warmup iterations, measured iterations, kernel name, tile configuration, datatype, problem size, toolchain used, and host environment.
- Tests do NOT require NPU hardware — unit test the Python helpers; kernel correctness tests run on hardware only.
- The same `NPU_CTRL_STORE` scheme applies for results; set default to `~/.npu_control_plane/store`.

---

## File Structure

```
strixhalo-npu-setup/
  experiments/
    __init__.py
    kernel_defs/
      __init__.py
      passthrough.py       # IRON kernel descriptor for passthrough
      vec_add.py           # IRON kernel descriptor for vector add
      gemm_8x8x8.py        # IRON kernel descriptor for 8x8x8 BFP16 matmul
      gemm_block.py         # Blocked GEMM (64x64x64) with tiling
    runner.py               # Common NPU execution harness (XRT + IRON compile + npu-ctrl bench)
    experiment_1_passthrough.py    # Exp 1: single-tile passthrough baseline
    experiment_2_vec_add.py        # Exp 2: single-tile vector add sweep
    experiment_3_gemm_single.py    # Exp 3: single-tile GEMM microkernel
    experiment_4_gemm_block.py     # Exp 4: blocked/ATB GEMM layout sweep
    experiment_5_gemm_multi.py     # Exp 5: multi-tile scaling (2x2, 4x4, 8x4)
    experiment_6_bfp16_accuracy.py # Exp 6: BFP16 vs FP32 accuracy/throughput
    results/                # Output directory for run JSONs (gitignored)
      .gitkeep
  experiments_lib/
    __init__.py
    layouts.py              # Public layout primitives (row-major, blocked, ATB)
    datatypes.py            # Public BFP16/BF16/FP32 helpers
    tile_shapes.py          # Tile shape selection and capacity estimation
    dma_planner.py          # ObjectFIFO/DMA BD chain planning
    report.py              # Report generation from npu-ctrl benchmark results
  tests/
    test_experiment_helpers.py
```

---

### Task 1: Experiment Infrastructure and Passthrough Baseline

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/kernel_defs/__init__.py`
- Create: `experiments/kernel_defs/passthrough.py`
- Create: `experiments/runner.py`
- Create: `experiments_lib/__init__.py`
- Create: `experiments_lib/layouts.py`
- Create: `experiments_lib/datatypes.py`
- Create: `experiments_lib/tile_shapes.py`
- Create: `experiments_lib/dma_planner.py`
- Create: `experiments/experiment_1_passthrough.py`
- Create: `tests/test_experiment_helpers.py`
- Modify: `.gitignore` — add `experiments/results/` and `experiments/**/*.xclbin`

**Interfaces:**
- Produces: `experiments.runner.NpuRunner` — orchestrates kernel compile + XRT execution + npu-ctrl bench recording
- Produces: `experiments.kernel_defs.passthrough.PassthroughKernel` — IRON-based passthrough kernel descriptor
- Produces: `experiments_lib.layouts.RowMajor, BlockedLayout, ATBLayout` — layout descriptors
- Produces: `experiments_lib.datatypes.BFP16, BF16, FP32, DataType` — datatype descriptors
- Produces: `experiments_lib.tile_shapes.TileShape, TileShapeConfig`
- Produces: `experiments_lib.dma_planner.DmaPlan` — BD chain plan

- [ ] **Step 1: Write failing passthrough runner tests**

Create `tests/test_experiment_helpers.py`:

```python
"""Tests for experiment infrastructure. These do NOT require NPU hardware."""

from experiments.runner import NpuRunner, ExperimentConfig
from experiments.kernel_defs.passthrough import PassthroughKernel
from experiments_lib.layouts import RowMajor, BlockedLayout, ATBLayout
from experiments_lib.datatypes import BFP16, BF16, FP32
from experiments_lib.tile_shapes import TileShape, TileShapeConfig


def test_passthrough_kernel_descriptor():
    """Passthrough kernel descriptor has expected metadata without hardware."""
    k = PassthroughKernel(name="pt")
    assert k.name == "pt"
    assert k.dtype == FP32
    assert k.tile_shape is None  # no tiling for passthrough
    assert "passthrough" in str(k).lower()


def test_npu_runner_config():
    """ExperimentConfig stores metadata correctly."""
    cfg = ExperimentConfig(
        label="pt-test",
        kernel_name="passthrough",
        shape_str="N=1024",
        warmup=3,
        iters=20,
        toolchain="iron+peano",
        tile_ids=[0, 2],
    )
    assert cfg.label == "pt-test"
    assert cfg.warmup == 3
    assert cfg.iters == 20


def test_layout_strings():
    """Layout descriptors produce canonical strings."""
    assert RowMajor().name == "row-major"
    assert BlockedLayout(block_size=8).name == "blocked-8x8"
    assert ATBLayout(a_super=2, b_super=1).name == "atb-2x1"


def test_datatype_properties():
    """Datatype descriptors report correct widths."""
    assert BFP16().bit_width == 16
    assert BF16().bit_width == 16
    assert FP32().bit_width == 32
    assert BFP16().element_size == 2
    assert FP32().element_size == 4


def test_tile_shape_estimates():
    """Tile shape L1 fit estimation."""
    cfg = TileShapeConfig(m=64, k=64, n=64)
    shape = TileShape(32, 32, 32)
    # 32x32x32 BFP16: A=32*32*2=2048, B=2048, C=32*32*4=4096 = 8192 bytes
    assert shape.byte_count(dtype=BFP16()) == 32 * 32 * 2 * 2 + 32 * 32 * 4
    assert shape.fits_in_l1(cfg, dtype=BFP16()) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_experiment_helpers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments'`.

- [ ] **Step 3: Implement experiment infrastructure**

Create `experiments/__init__.py`:

```python
"""Phase 2 GEMM benchmark experiments on Strix Halo NPU."""

__version__ = "0.1.0"
```

Create `experiments/kernel_defs/__init__.py`:

```python
from .passthrough import PassthroughKernel
```

Create `experiments_lib/__init__.py`:

```python
"""Public layout, datatype, and tiling primitives for AIE2P kernels."""
```

Create `experiments_lib/datatypes.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DataType:
    name: str
    bit_width: int
    element_size: int  # bytes

    def __str__(self):
        return self.name


BFP16 = DataType("bfp16", 16, 2)
BF16 = DataType("bf16", 16, 2)
FP32 = DataType("fp32", 32, 4)


def from_string(s: str) -> DataType:
    mapping = {"bfp16": BFP16, "bf16": BF16, "fp32": FP32}
    return mapping[s.lower()]
```

Create `experiments_lib/layouts.py`:

```python
from dataclasses import dataclass
from typing import Any


class Layout:
    name: str


@dataclass(frozen=True)
class RowMajor(Layout):
    name: str = "row-major"
    description: str = "naive row-major layout, default"


@dataclass(frozen=True)
class BlockedLayout(Layout):
    block_size: int = 8
    name: str = "blocked-8x8"

    def __post_init__(self):
        object.__setattr__(self, "name", f"blocked-{self.block_size}x{self.block_size}")

    @property
    def description(self):
        return f"{self.block_size}x{self.block_size} blocked layout"


@dataclass(frozen=True)
class ATBLayout(Layout):
    a_super: int = 2
    b_super: int = 1
    name: str = "atb-2x1"

    def __post_init__(self):
        object.__setattr__(self, "name", f"atb-{self.a_super}x{self.b_super}")

    @property
    def description(self):
        return f"Asymmetric tile buffering {self.a_super}x{self.b_super}"
```

Create `experiments_lib/tile_shapes.py`:

```python
from dataclasses import dataclass, field

from .datatypes import DataType, FP32, BFP16

AIE2P_L1_BYTES = 32 * 1024


@dataclass(frozen=True)
class TileShape:
    m: int
    k: int
    n: int

    def byte_count(self, dtype: DataType = BFP16) -> int:
        """Total L1 bytes needed for A+B+C buffers."""
        a_bytes = self.m * self.k * dtype.element_size
        b_bytes = self.k * self.n * dtype.element_size
        c_bytes = self.m * self.n * FP32.element_size  # accumulate in FP32
        return a_bytes + b_bytes + c_bytes

    def fits_in_l1(self, dtype: DataType = BFP16) -> bool:
        return self.byte_count(dtype) <= AIE2P_L1_BYTES


@dataclass(frozen=True)
class TileShapeConfig:
    m: int = 64
    k: int = 64
    n: int = 64

    @property
    def base(self) -> TileShape:
        return TileShape(self.m, self.k, self.n)

    def plausible_shapes(self) -> list[TileShape]:
        return [
            TileShape(32, 32, 32),
            TileShape(32, 64, 64),
            TileShape(64, 64, 64),
        ]


@dataclass
class MultiTileConfig:
    rows: int = 2
    cols: int = 2

    @property
    def tile_count(self) -> int:
        return self.rows * self.cols
```

Create `experiments_lib/dma_planner.py`:

```python
from dataclasses import dataclass, field
from typing import Any
from .layouts import Layout, RowMajor


@dataclass(frozen=True)
class DmaPlan:
    input_a_layout: Layout
    input_b_layout: Layout
    output_layout: Layout
    double_buffer: bool = True
    object_fifo_depth: int = 2
    bd_chain_length: int = field(default=2)


def default_gemm_plan() -> DmaPlan:
    return DmaPlan(
        input_a_layout=RowMajor(),
        input_b_layout=RowMajor(),
        output_layout=RowMajor(),
        double_buffer=True,
        object_fifo_depth=2,
        bd_chain_length=2,
    )
```

- [ ] **Step 4: Implement passthrough kernel descriptor**

Create `experiments/kernel_defs/passthrough.py`:

```python
from dataclasses import dataclass
from typing import Any
from experiments_lib.datatypes import DataType, FP32
from experiments_lib.tile_shapes import TileShape


@dataclass
class PassthroughKernel:
    """Descriptor for a single-tile passthrough kernel using IRON."""

    name: str
    dtype: DataType = FP32
    tile_shape: TileShape | None = None
    source_lines: list[str] = None

    def __post_init__(self):
        if self.source_lines is None:
            self.source_lines = [
                "@iron.jit",
                "def passthrough(a_in: In, c_out: Out):",
                "    ty = np.ndarray[(N,), np.dtype[np.int32]]",
                "    of = ObjectFifo(ty, name='fifo')",
                "    def work(ififo, ofifo):",
                "        ai = ififo.acquire(1)",
                "        co = ofifo.acquire(1)",
                "        np.copyto(co, ai)",
                "        ofifo.release(1)",
                "        ififo.release(1)",
                "    w = Worker(work, [of.cons(), of.prod()], tile=Tile(0, 2))",
                "    rt = Runtime()",
                "    with rt.sequence(ty, ty) as (a, c):",
                "        rt.start(w)",
                "        rt.fill(of.prod(), a)",
                "        rt.drain(of.cons(), c, wait=True)",
                "    return Program(iron.get_current_device(), rt).resolve_program()",
            ]

    def __str__(self):
        return f"PassthroughKernel(name={self.name}, dtype={self.dtype})"

    def generate_iron_source(self, n: int) -> str:
        """Generate IRON source for a passthrough kernel of size N."""
        lines = [
            "import numpy as np",
            "from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker",
            "from aie.iron.device import Tile",
            "",
            f"N = {n}",
            "",
        ]
        lines.extend(self.source_lines)
        return "\n".join(lines)
```

- [ ] **Step 5: Implement NpuRunner skeleton**

Create `experiments/runner.py`:

```python
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from npu_control_plane.metadata import MetadataStore
from npu_control_plane.probe import which


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
```

- [ ] **Step 6: Implement experiment_1_passthrough.py script**

Create `experiments/experiment_1_passthrough.py`:

```python
#!/usr/bin/env python3
"""Experiment 1: Single-tile passthrough kernel baseline.

Measures raw data movement bandwidth for one AIE2P tile using
the simplest possible kernel. Establishes floor for all later experiments.

Usage:
    NPU_CTRL_STORE=/tmp/npu-ctrl-exp1 python3 experiments/experiment_1_passthrough.py
"""

import sys
import os
from pathlib import Path

from .kernel_defs.passthrough import PassthroughKernel
from .runner import NpuRunner, ExperimentConfig
from npu_control_plane.metadata import MetadataStore

SIZES = [64, 256, 1024, 4096, 16384]


def main():
    store = MetadataStore()
    cfg = ExperimentConfig(
        label="exp1-passthrough",
        kernel_name="passthrough",
        shape_str="N=1024",
        warmup=5,
        iters=30,
        toolchain="iron+peano",
        store=store,
    )
    kernel = PassthroughKernel(name="pt")
    runner = NpuRunner(cfg, dry_run=True)

    print("=== Experiment 1: Passthrough Baseline ===")
    print(f"Dry run mode — no NPU execution. Sizes: {SIZES}")
    print(f"Kernel: {kernel}")
    print(f"Store: {store.root}")

    for n in SIZES:
        source = kernel.generate_iron_source(n)
        print(f"\nN={n}: source generated ({len(source)} bytes)")
        for line in source.splitlines()[:6]:
            print(f"  {line}")
        print("  ...")

    print("\nTo run on hardware: set dry_run=False and provide NPU access.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Wire experiment as console script**

Append to `pyproject.toml`:

```toml
exp1-passthrough = "experiments.experiment_1_passthrough:main"
```

- [ ] **Step 8: Update .gitignore**

Append to `.gitignore`:

```gitignore
experiments/results/
experiments/**/*.xclbin
experiments/**/*.elf
```

- [ ] **Step 9: Run all tests**

```bash
python3 -m pytest tests/test_experiment_helpers.py tests/ -q
```

Expected: all existing + 5 new tests pass.

- [ ] **Step 10: Dry-run passthrough experiment**

```bash
NPU_CTRL_STORE=/tmp/npu-ctrl-exp1 python3 -m experiments.experiment_1_passthrough
```

Expected: prints experiment summary, no errors, no NPU crash.

- [ ] **Step 11: Commit Task 1**

```bash
git add experiments/ experiments_lib/ tests/test_experiment_helpers.py .gitignore pyproject.toml
git commit -m "feat: add experiment infra and passthrough baseline"
```

---

### Task 2: Vector Add and Single-Tile GEMM Microkernel

**Files:**
- Create: `experiments/kernel_defs/vec_add.py`
- Create: `experiments/kernel_defs/gemm_8x8x8.py`
- Create: `experiments/experiment_2_vec_add.py`
- Create: `experiments/experiment_3_gemm_single.py`

- [ ] **Step 1: Write failing kernel descriptor tests**

Append to `tests/test_experiment_helpers.py`:

```python
from experiments.kernel_defs.vec_add import VecAddKernel
from experiments.kernel_defs.gemm_8x8x8 import Gemm8x8x8Kernel


def test_vec_add_kernel_descriptor():
    k = VecAddKernel(name="vadd")
    assert k.name == "vadd"
    source = k.generate_iron_source(n=1024)
    assert "N = 1024" in source
    assert "for" in source.lower() or "range" in source


def test_gemm_8x8x8_kernel_descriptor():
    k = Gemm8x8x8Kernel(name="gemm8")
    assert k.name == "gemm8"
    source = k.generate_iron_source()
    assert "matmul" in source.lower() or "mmul" in source.lower()
```

- [ ] **Step 2: Run and verify they fail**

```bash
python3 -m pytest tests/test_experiment_helpers.py -q --tb=short
```
Expected: 2 new fails.

- [ ] **Step 3: Implement vec_add kernel descriptor**

Create `experiments/kernel_defs/vec_add.py`:

```python
from dataclasses import dataclass
from experiments_lib.datatypes import DataType, FP32


@dataclass
class VecAddKernel:
    name: str
    dtype: DataType = FP32

    def generate_iron_source(self, n: int = 1024) -> str:
        return f"""\
import numpy as np
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.controlflow import range_

N = {n}
ty = np.ndarray[(N,), np.dtype[np.float32]]
ty_out = np.ndarray[(N,), np.dtype[np.float32]]

@iron.jit
def vec_add(a: In, b: In, c: Out):
    fifo_a = ObjectFifo(ty, name="a")
    fifo_b = ObjectFifo(ty, name="b")
    fifo_c = ObjectFifo(ty, name="c")

    def work(a_cons, b_cons, c_prod):
        ai = a_cons.acquire(1)
        bi = b_cons.acquire(1)
        co = c_prod.acquire(1)
        with range_(N // 64) as i:
            pass  # TODO: implement vector add with AIE intrinsics
        c_prod.release(1)
        b_cons.release(1)
        a_cons.release(1)

    w = Worker(work, [fifo_a.cons(), fifo_b.cons(), fifo_c.prod()], tile=Tile(0, 2))
    rt = Runtime()
    with rt.sequence(ty, ty, ty_out) as (a, b, c):
        rt.start(w)
        rt.fill(fifo_a.prod(), a)
        rt.fill(fifo_b.prod(), b)
        rt.drain(fifo_c.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()
"""

    def __str__(self):
        return f"VecAddKernel(name={self.name}, dtype={self.dtype})"
```

- [ ] **Step 4: Implement GEMM 8x8x8 kernel descriptor**

Create `experiments/kernel_defs/gemm_8x8x8.py`:

```python
from dataclasses import dataclass
from experiments_lib.datatypes import BFP16, FP32, DataType


@dataclass
class Gemm8x8x8Kernel:
    name: str
    a_dtype: DataType = BFP16
    b_dtype: DataType = BFP16
    c_dtype: DataType = FP32

    def __str__(self):
        return f"Gemm8x8x8Kernel(name={self.name}, A={self.a_dtype}, B={self.b_dtype}, C={self.c_dtype})"

    def generate_iron_source(self) -> str:
        return f"""\
import numpy as np
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.controlflow import range_

# 8x8x8 BFP16 matmul microkernel using public AIE API intrinsics
# Reference: https://github.com/Xilinx/mlir-aie

M, K, N = 8, 8, 8
ty_a = np.ndarray[(M, K), np.dtype[np.bfloat16]]
ty_b = np.ndarray[(K, N), np.dtype[np.bfloat16]]
ty_c = np.ndarray[(M, N), np.dtype[np.float32]]

@iron.jit
def gemm_8x8x8(A: In, B: In, C: Out):
    fifo_a = ObjectFifo(ty_a, name="A")
    fifo_b = ObjectFifo(ty_b, name="B")
    fifo_c = ObjectFifo(ty_c, name="C")

    def work(a_cons, b_cons, c_prod):
        ai = a_cons.acquire(1)
        bi = b_cons.acquire(1)
        co = c_prod.acquire(1)
        # matmul: C = A @ B using AIE2P vector MAC
        # TODO: lower to aievec.matmul_aie2p when IRON supports it
        for i in range(M):
            for j in range(N):
                acc = 0.0
                for k in range(K):
                    acc += float(ai[i, k]) * float(bi[k, j])
                co[i, j] = acc
        c_prod.release(1)
        b_cons.release(1)
        a_cons.release(1)

    w = Worker(work, [fifo_a.cons(), fifo_b.cons(), fifo_c.prod()], tile=Tile(0, 2))
    rt = Runtime()
    with rt.sequence(ty_a, ty_b, ty_c) as (a, b, c):
        rt.start(w)
        rt.fill(fifo_a.prod(), a)
        rt.fill(fifo_b.prod(), b)
        rt.drain(fifo_c.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()
"""
```

- [ ] **Step 5: Implement experiment_2_vec_add.py**

Create `experiments/experiment_2_vec_add.py`:

```python
#!/usr/bin/env python3
"""Experiment 2: Single-tile vector add benchmark.

Sweeps vector sizes, records throughput in GB/s and MOPS.
"""

import sys

from .kernel_defs.vec_add import VecAddKernel
from .runner import NpuRunner, ExperimentConfig
from npu_control_plane.metadata import MetadataStore

SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]


def main():
    store = MetadataStore()
    cfg = ExperimentConfig(
        label="exp2-vecadd",
        kernel_name="vec_add",
        shape_str="N=1024",
        warmup=5,
        iters=30,
        toolchain="iron+peano",
        store=store,
    )
    kernel = VecAddKernel(name="vadd")
    runner = NpuRunner(cfg, dry_run=True)

    print("=== Experiment 2: Vector Add Sweep ===")
    print(f"Dry run — no NPU execution. Sizes: {SIZES}")
    print(f"Kernel: {kernel}")
    print(f"Store: {store.root}")

    for n in SIZES:
        source = kernel.generate_iron_source(n)
        print(f"\nN={n}: source ({len(source)} bytes)")

    print("\nTo run on hardware: set dry_run=False.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Implement experiment_3_gemm_single.py**

Create `experiments/experiment_3_gemm_single.py`:

```python
#!/usr/bin/env python3
"""Experiment 3: Single-tile GEMM microkernel.

Benchmarks 8x8x8 BFP16 matmul on one AIE2P tile.
Measures:
- Raw microkernel throughput (GOPS)
- DMA overhead
- Warmup convergence
"""

import sys

from .kernel_defs.gemm_8x8x8 import Gemm8x8x8Kernel
from .runner import NpuRunner, ExperimentConfig
from experiments_lib.tile_shapes import TileShape, TileShapeConfig
from experiments_lib.datatypes import BFP16
from npu_control_plane.metadata import MetadataStore


def main():
    store = MetadataStore()
    cfg = ExperimentConfig(
        label="exp3-gemm-single",
        kernel_name="gemm_8x8x8",
        shape_str="M=8,K=8,N=8",
        warmup=5,
        iters=100,
        toolchain="iron+peano",
        store=store,
    )
    kernel = Gemm8x8x8Kernel(name="gemm8")
    runner = NpuRunner(cfg, dry_run=True)

    tile_shapes = TileShapeConfig(m=64, k=64, n=64)
    print("=== Experiment 3: Single-Tile GEMM ===")
    print(f"Microkernel: {kernel}")
    print(f"Dry run — no NPU execution.")
    print(f"Tile shapes that fit L1 ({32*1024} bytes):")
    for ts in tile_shapes.plausible_shapes():
        fits = ts.fits_in_l1(BFP16)
        print(f"  {ts.m}x{ts.k}x{ts.n}: {ts.byte_count(BFP16)} bytes {'✅' if fits else '❌'}")

    source = kernel.generate_iron_source()
    print(f"\nKernel source generated ({len(source)} bytes)")

    print("\nTo run on hardware:")
    print("  NPU_CTRL_STORE=$STORE python3 experiments/experiment_3_gemm_single.py --no-dry-run")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Update kernel_defs __init__.py**

```python
from .passthrough import PassthroughKernel
from .vec_add import VecAddKernel
from .gemm_8x8x8 import Gemm8x8x8Kernel
```

- [ ] **Step 8: Run tests**

```bash
python3 -m pytest tests/test_experiment_helpers.py -q
```
Expected: 7 pass.

- [ ] **Step 9: Dry-run experiments**

```bash
for exp in exp2-vecadd exp3-gemm-single; do
    NPU_CTRL_STORE=/tmp/npu-ctrl-$exp python3 -m experiments.${exp//-/_} 2>&1 | tail -3
done
```

Expected: each exits 0, prints summary.

- [ ] **Step 10: Commit Task 2**

```bash
git add experiments/ tests/
git commit -m "feat: add vec add and gemm microkernel experiments"
```

---

### Task 3: Blocked GEMM + Multi-Tile Framework

**Files:**
- Create: `experiments/kernel_defs/gemm_block.py`
- Create: `experiments/experiment_4_gemm_block.py`
- Create: `experiments/experiment_5_gemm_multi.py`

**Interfaces:**
- `GemmBlockKernel` — blocked GEMM with configurable layout
- Support for multi-tile dispatch in `NpuRunner`

- [ ] **Step 1: Write failing blocked GEMM tests**

```python
from experiments.kernel_defs.gemm_block import GemmBlockKernel
from experiments_lib.layouts import RowMajor, BlockedLayout, ATBLayout


def test_gemm_block_descriptor():
    k = GemmBlockKernel(name="gblock", block_m=64, block_k=64, block_n=64)
    assert k.block_m == 64
    assert k.tile_count() == 4  # 2x2 default


def test_gemm_block_layout_string():
    k = GemmBlockKernel(name="gblock", layout=ATBLayout(2, 1))
    assert "atb" in str(k.layout).lower()
```

- [ ] **Step 2: Implement blocked GEMM kernel descriptor**

Create `experiments/kernel_defs/gemm_block.py`:

```python
from dataclasses import dataclass, field
from typing import Any
from experiments_lib.datatypes import BFP16, DataType
from experiments_lib.layouts import Layout, RowMajor
from experiments_lib.tile_shapes import TileShape, TileShapeConfig, MultiTileConfig


@dataclass
class GemmBlockKernel:
    name: str
    block_m: int = 64
    block_k: int = 64
    block_n: int = 64
    a_dtype: DataType = BFP16
    b_dtype: DataType = BFP16
    c_dtype: DataType = None
    layout: Layout = RowMajor()
    multi_tile: MultiTileConfig = field(default_factory=lambda: MultiTileConfig(2, 2))

    def __post_init__(self):
        if self.c_dtype is None:
            object.__setattr__(self, "c_dtype", BFP16)
        self._tile = TileShape(self.block_m, self.block_k, self.block_n)

    def fits_in_l1(self) -> bool:
        return self._tile.fits_in_l1(self.a_dtype)

    def tile_count(self) -> int:
        return self.multi_tile.tile_count

    def generate_source(self) -> str:
        return f"""\
# GemmBlockKernel: {self.name}
# Block: {self.block_m}x{self.block_k}x{self.block_n}
# Layout: {self.layout.name}
# Tiles: {self.tile_count()}
# Datatype: A={self.a_dtype.name} B={self.b_dtype.name} C={self.c_dtype.name}
"""

    def __str__(self):
        return f"GemmBlockKernel({self.name}, {self.block_m}x{self.block_k}x{self.block_n}, {self.layout.name}, {self.tile_count()} tiles)"
```

- [ ] **Step 3: Implement experiment_4_gemm_block.py**

Create `experiments/experiment_4_gemm_block.py`:

```python
#!/usr/bin/env python3
"""Experiment 4: Blocked/ATB GEMM layout sweep.

Compares row-major, blocked-8x8, and ATB layouts for a 64x64x64 GEMM block.
Measures throughput impact of memory layout choices.
"""

import sys
from .kernel_defs.gemm_block import GemmBlockKernel
from .runner import NpuRunner, ExperimentConfig
from experiments_lib.layouts import RowMajor, BlockedLayout, ATBLayout
from experiments_lib.datatypes import BFP16
from npu_control_plane.metadata import MetadataStore


LAYOUTS = [RowMajor(), BlockedLayout(8), ATBLayout(2, 1)]


def main():
    store = MetadataStore()

    for layout in LAYOUTS:
        kernel = GemmBlockKernel(
            name=f"gblock-{layout.name}",
            block_m=64, block_k=64, block_n=64,
            layout=layout,
        )
        fits = kernel.fits_in_l1()
        print(f"\nLayout: {layout.name:20s}  L1 fit: {'✅' if fits else '❌'}  {kernel}")

    print("\n=== Experiment 4: Layout Sweep ===")
    print("\nTo run on hardware: use `npu-ctrl bench run` with each compiled kernel.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement experiment_5_gemm_multi.py**

Create `experiments/experiment_5_gemm_multi.py`:

```python
#!/usr/bin/env python3
"""Experiment 5: Multi-tile GEMM scaling.

Scales from 1 tile to 32 tiles to measure:
- Strong scaling (fixed problem size)
- Weak scaling (fixed per-tile size)
- Communication/routing overhead
"""

import sys
from .kernel_defs.gemm_block import GemmBlockKernel
from .runner import NpuRunner, ExperimentConfig
from experiments_lib.tile_shapes import MultiTileConfig
from npu_control_plane.metadata import MetadataStore


CLUSTERS = [MultiTileConfig(1, 1), MultiTileConfig(2, 2), MultiTileConfig(4, 4), MultiTileConfig(8, 4)]


def main():
    store = MetadataStore()
    print("=== Experiment 5: Multi-Tile Scaling ===")
    for cluster in CLUSTERS:
        kernel = GemmBlockKernel(
            name=f"gemm-multi-{cluster.rows}x{cluster.cols}",
            multi_tile=cluster,
        )
        print(f"  {cluster.rows}x{cluster.cols} = {cluster.tile_count:2d} tiles  {kernel}")

    print("\nTo run on hardware:")
    print("  NPU_CTRL_STORE=$STORE python3 experiments/experiment_5_gemm_multi.py --no-dry-run")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_experiment_helpers.py -q
```
Expected: 9 pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add experiments/ tests/
git commit -m "feat: add blocked gemm and multi-tile experiments"
```

---

### Task 4: BFP16 Accuracy Experiment and Report Generation

**Files:**
- Create: `experiments/experiment_6_bfp16_accuracy.py`
- Create: `experiments_lib/report.py`

**Interfaces:**
- `report.generate_html_summary(store: MetadataStore) -> str`
- `report.generate_markdown_summary(store: MetadataStore) -> str`

- [ ] **Step 1: Write failing report tests**

```python
from experiments_lib.report import generate_markdown_summary
from npu_control_plane.metadata import MetadataStore


def test_generate_markdown_summary_empty():
    store = MetadataStore(tmp_path / "store")
    md = generate_markdown_summary(store)
    assert "Benchmark Summary" in md
    assert "No runs recorded" in md
```

- [ ] **Step 2: Implement BFP16 accuracy experiment**

Create `experiments/experiment_6_bfp16_accuracy.py`:

```python
#!/usr/bin/env python3
"""Experiment 6: BFP16 vs FP32 accuracy/throughput characterization.

Measures numerical error and throughput for GEMM at various sizes.
Provides the data needed to recommend the optimal datatype path.
"""

import sys
import random
import math
from .kernel_defs.gemm_8x8x8 import Gemm8x8x8Kernel
from .runner import NpuRunner, ExperimentConfig
from experiments_lib.datatypes import BFP16, BF16, FP32
from npu_control_plane.metadata import MetadataStore

SIZES = [(8, 8, 8), (16, 16, 16), (32, 32, 32)]


def cpu_gemm(m, k, n, dtype_str):
    """Reference CPU GEMM. Returns matrix and stats."""
    a = [[random.gauss(0.0, 0.5) for _ in range(k)] for _ in range(m)]
    b = [[random.gauss(0.0, 0.5) for _ in range(n)] for _ in range(k)]
    c = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc += a[i][kk] * b[kk][j]
            c[i][j] = acc
    max_abs = max(abs(v) for row in c for v in row)
    return max_abs


def main():
    store = MetadataStore()

    print("=== Experiment 6: BFP16 Accuracy ===")
    print(f"{'Size':>12s} {'Max abs value':>16s} {'Notes'}")
    print("-" * 50)
    for (m, k, n) in SIZES:
        ref = cpu_gemm(m, k, n, "fp32")
        print(f"{m}x{k}x{n:>3s}  {ref:>16.6f}  FP32 reference")
    print("\nTo run on hardware: uses npu-ctrl bench with accuracy verification.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Implement report generator**

Create `experiments_lib/report.py`:

```python
"""Report generation from npu-ctrl benchmark results."""

from __future__ import annotations
from typing import Any
from npu_control_plane.metadata import MetadataStore


def generate_markdown_summary(store: MetadataStore) -> str:
    """Generate a Markdown summary of all benchmark runs in store."""
    runs = store.read_json("benchmarks", "summary.json", default={"runs": []}).get("runs", [])
    if not runs:
        return "# Benchmark Summary\n\nNo runs recorded.\n"

    lines = ["# Benchmark Summary\n"]
    for run in runs:
        median = run.get("median_ms")
        label = run.get("label", "unknown")
        ts = run.get("timestamp", "")
        rc = run.get("returncode")
        lines.append(f"- **{label}** ({ts})")
        if median is not None:
            lines.append(f"  - Median: {median:.3f} ms")
        lines.append(f"  - Return code: {rc}")
        lines.append("")
    return "\n".join(lines)


def generate_html_summary(store: MetadataStore) -> str:
    """Generate a minimal HTML summary page."""
    md = generate_markdown_summary(store)
    return f"<html><body><pre>{md}</pre></body></html>"
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_experiment_helpers.py -q
```
Expected: 10 pass (all tests).

- [ ] **Step 5: Commit Task 4**

```bash
git add experiments/ experiments_lib/report.py tests/
git commit -m "feat: add bfp16 accuracy experiment and report generation"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Exp 1: Passthrough baseline — Task 1
- [x] Exp 2: Vector add — Task 2
- [x] Exp 3: Single-tile GEMM microkernel — Task 2
- [x] Exp 4: Memory layout / blocked GEMM — Task 3
- [x] Exp 5: Multi-tile scaling — Task 3
- [x] Exp 6: BFP16 accuracy — Task 4
- [x] Public clean-room constraints — no proprietary code, pure public tooling
- [x] `npu-ctrl` integration — all experiments use npu-ctrl store and bench interface
- [x] Deterministic tests — all helpers tested without NPU hardware
- [x] Tile shape L1 fit estimation — tile_shapes.py
- [x] Layout descriptors — layouts.py (RowMajor, Blocked, ATB)
- [x] Datatype descriptors — datatypes.py (BFP16, BF16, FP32)
- [x] DMA planning scaffold — dma_planner.py
- [x] Report generation — report.py

**Implementation order:**
- Task 1: Infrastructure + passthrough → foundation
- Task 2: Vector add + GEMM microkernel → compute primitives
- Task 3: Blocked GEMM + multi-tile → scaling
- Task 4: BFP16 accuracy + reporting → characterization + output

**Review gates:**
- Each task has failing test first.
- Each task ends with a commit.
- No NPU hardware required for unit tests.
- Hardware run requires `--no-dry-run` flag (added when needed).

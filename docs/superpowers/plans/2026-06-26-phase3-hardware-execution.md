# Phase 3: Hardware Compilation and Execution

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up real IRON JIT + Peano compilation in NpuRunner and run the first passthrough kernel on actual Strix Halo NPU hardware.

**Architecture:** Fix kernel descriptors to produce valid IRON JIT Python code, add a real compilation/execution path in NpuRunner using IRON's `@iron.jit` decorator, test compilation with `pyxrt` runtime verification, and record benchmark results via `npu-ctrl bench run`.

**Tech Stack:** Python 3.14, `aie.iron` (MLIR-AIE), Peano/LLVM-AIE (`clang++ --target=aie2`), XRT (`pyxrt`), `npu-ctrl` CLI.

## Global Constraints

- Public/open-source clean-room: only IRON (Apache 2.0) and Peano (Apache 2.0 w/ LLVM exceptions).
- All kernel source must be hand-written, not derived from proprietary Chess output.
- Tests that require NPU hardware must be explicitly gated with a `--hardware` flag or `NPU_AVAILABLE` env check.
- Use the npu-ctrl metadata store for all result recording.
- Peano path: `$PEANO_INSTALL_DIR` or `/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie/bin/clang++`.

---

### Task 1: Fix Passthrough Kernel to Compile via IRON JIT

**Files:**
- Modify: `experiments/kernel_defs/passthrough.py` — replace illustrative source with real compilable IRON JIT code
- Modify: `experiments/runner.py` — wire `NpuRunner.compile_kernel()` to invoke IRON JIT
- Create: `experiments/hardware_test_runner.py` — script that runs compilation path with hardware check

- [ ] **Step 1: Inspect a known-working IRON JIT example**

```bash
source /home/bcloud/mlir-aie/.venv/bin/activate
cd /home/bcloud/mlir-aie
ls programming_examples/basic/passthrough/
cat programming_examples/basic/passthrough/*.py | head -60
```

- [ ] **Step 2: Write a test passthrough kernel directly**

Create `/tmp/test_iron_pt.py`:

```python
import numpy as np
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile

N = 64
ty = np.ndarray[(N,), np.dtype[np.int32]]

@iron.jit
def test_pt(a: In, c: Out):
    of = ObjectFifo(ty, name="fifo")
    def work(cons, prod):
        ai = cons.acquire(1)
        co = prod.acquire(1)
        np.copyto(co, ai)
        prod.release(1)
        cons.release(1)
    w = Worker(work, [of.cons(), of.prod()], tile=Tile(0, 2))
    rt = Runtime()
    with rt.sequence(ty, ty) as (a, c):
        rt.start(w)
        rt.fill(of.prod(), a)
        rt.drain(of.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()
```

- [ ] **Step 3: Test compilation**

```bash
source /home/bcloud/mlir-aie/.venv/bin/activate
python3 /tmp/test_iron_pt.py
```

Expected: IRON JIT compiles the kernel. If it succeeds, record the compilation output, the artifact path, the runtime version.

- [ ] **Step 4: Update PassthroughKernel to produce real IRON source**

Replace `passthrough.py` source_lines with dynamically generated valid IRON code:

```python
def generate_iron_source(self, n: int = 64, dtype_str: str = "int32") -> str:
    return f"""\
import numpy as np
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile

N = {n}
ty = np.ndarray[(N,), np.dtype[np.{dtype_str}]]

@iron.jit
def passthrough(a: In, c: Out):
    of = ObjectFifo(ty, name="fifo")
    def work(cons, prod):
        ai = cons.acquire(1)
        co = prod.acquire(1)
        np.copyto(co, ai)
        prod.release(1)
        cons.release(1)
    w = Worker(work, [of.cons(), of.prod()], tile=Tile(0, 2))
    rt = Runtime()
    with rt.sequence(ty, ty) as (a, c):
        rt.start(w)
        rt.fill(of.prod(), a)
        rt.drain(of.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()
"""
```

- [ ] **Step 5: Wire NpuRunner.compile_kernel()**

Replace the `NotImplementedError` in `NpuRunner.compile_kernel()`:

```python
def compile_kernel(self, source: str, tmp_dir: Path | None = None) -> Path:
    """Write IRON source and run it through the JIT, which returns a .xclbin or compiled artifact."""
    tmp = tmp_dir or Path(tempfile.mkdtemp())
    src_path = tmp / f"{self.config.name}.py"
    src_path.write_text(source)

    result = run_command(
        [sys.executable, "-m", "py_compile", str(src_path)],
        timeout=30,
    )
    # Actually, IRON JIT doesn't work via py_compile. We need to run the script.
    # The IRON JIT compiles at import/execution time and writes artifacts.
    # Return the source path; execution is separate.
    return src_path
```

Actually, IRON JIT works differently. The `@iron.jit` decorator compiles the function when it's first called. The compiled artifact (xclbin/ELF) is cached. So we need to:
1. Generate the source
2. Write it to a temp file
3. Execute it with `subprocess.run([python, script])` — this triggers JIT compilation
4. The script should output measurements to stdout and return the artifact path

Let me update the approach:

```python
def compile_and_run(self, source: str) -> dict:
    """Write source to temp file, execute it via IRON JIT, return timing."""
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "kernel.py"
    src.write_text(source)
    result = run_command([sys.executable, str(src)], timeout=300)
    # Script outputs JSON timing record to stdout
    return json.loads(result.stdout)
```

- [ ] **Step 6: Add IRON peano/clang path detection**

```python
def detect_peano() -> str | None:
    """Return path to Peano clang++ or None."""
    candidates = [
        "/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie/bin/clang++",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None
```

- [ ] **Step 7: Run compilation smoke test**

```bash
source /home/bcloud/mlir-aie/.venv/bin/activate
cd /home/bcloud/strixhalo-npu-setup
python3 -c "
from experiments.kernel_defs.passthrough import PassthroughKernel
k = PassthroughKernel(name='pt')
src = k.generate_iron_source(n=64)
with open('/tmp/test_pt_gen.py', 'w') as f:
    f.write(src)
print('Generated', len(src), 'bytes')
"
```

- [ ] **Step 8: Commit Task 1**

```bash
git add experiments/
git commit -m "feat: real iron jit passthrough kernel"
```

---

### Task 2: Fix VecAdd and GEMM Microkernel for Real Compilation

**Files:**
- Modify: `experiments/kernel_defs/vec_add.py` — real IRON vector add
- Modify: `experiments/kernel_defs/gemm_8x8x8.py` — real IRON matmul
- Create: `experiments/kernel_defs/peano_helpers.py` — Peano clang invocation helpers

- [ ] **Step 1: Study AIE2P vector examples**

```bash
source /home/bcloud/mlir-aie/.venv/bin/activate
cat /home/bcloud/mlir-aie/programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering/*.py | head -100
cat /home/bcloud/mlir-aie/test/Conversion/AIEVecToLLVM/matmul-aie2p.mlir | head -60
```

- [ ] **Step 2: Implement real vector add kernel**

```python
# experiments/kernel_defs/vec_add.py — real IRON kernel
# Uses AIE2P vector add via public IRON API
```

- [ ] **Step 3: Implement real GEMM microkernel**

```python
# experiments/kernel_defs/gemm_8x8x8.py — real IRON matmul
# Uses public MLIR-AIE matmul examples as reference
```

- [ ] **Step 4: Add Peano compile helpers**

```python
# experiments/kernel_defs/peano_helpers.py
# Wraps Peano clang++ invocation for external C++ AIE kernels
```

- [ ] **Step 5: Run compilation tests**

```bash
source /home/bcloud/mlir-aie/.venv/bin/activate
python3 experiments/kernel_defs/vec_add.py  # triggers JIT compilation
```

- [ ] **Step 6: Commit Task 2**

---

### Task 3: Run Experiment 1-3 on Real Hardware

**Files:**
- Create: `experiments/run_experiments.sh` — batch runner for all experiments
- Modify: Each experiment script to support `--real` flag for hardware execution

- [ ] **Step 1: Add --real flag to experiment scripts**

```python
# argparse: --real flag enables actual NPU execution
```

- [ ] **Step 2: Run passthrough on hardware**

```bash
source /home/bcloud/mlir-aie/.venv/bin/activate
export NPU_CTRL_STORE=/tmp/npu-ctrl-phase3
python3 experiments/experiment_1_passthrough.py --real --warmup 5 --iters 30
```

- [ ] **Step 3: Run vec add on hardware**

```bash
python3 experiments/experiment_2_vec_add.py --real --warmup 5 --iters 30
```

- [ ] **Step 4: Run GEMM microkernel on hardware**

```bash
python3 experiments/experiment_3_gemm_single.py --real --warmup 5 --iters 100
```

- [ ] **Step 5: Generate benchmark report**

```bash
python3 -c "
from experiments_lib.report import generate_markdown_summary
from npu_control_plane.metadata import MetadataStore
md = generate_markdown_summary(MetadataStore())
print(md)
"
```

- [ ] **Step 6: Commit results**

---

### Task 4: Run Experiment 4-6 on Real Hardware

**Files:**
- Same experiment scripts, now with layout/multi-tile/accuracy passes

- [ ] **Step 1: Run layout sweep**

```bash
python3 experiments/experiment_4_gemm_block.py --real
```

- [ ] **Step 2: Run multi-tile scaling**

```bash
python3 experiments/experiment_5_gemm_multi.py --real
```

- [ ] **Step 3: Run accuracy characterization**

```bash
python3 experiments/experiment_6_bfp16_accuracy.py --real
```

- [ ] **Step 4: Generate final report**

```bash
python3 -c "
from experiments_lib.report import generate_html_summary
from npu_control_plane.metadata import MetadataStore
html = generate_html_summary(MetadataStore())
Path('/tmp/npu-benchmark-report.html').write_text(html)
print('Report: /tmp/npu-benchmark-report.html')
"
```

- [ ] **Step 5: Commit results**

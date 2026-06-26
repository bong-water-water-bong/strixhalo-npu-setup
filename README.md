# Strix Halo NPU Setup

Step-by-step guide to unlock the NPU on **AMD Ryzen AI Max+ 395 (Strix Halo)** — from zero to **31 TFLOPS** AI compute.

## System

| Component | Spec |
|---|---|
| CPU | AMD RYZEN AI MAX+ 395 (16 Zen5 cores) |
| GPU | Radeon 8060S (40 RDNA 3.5 CUs) |
| NPU | RyzenAI-npu5 (32 AIE2P tiles) |
| RAM | 128 GB LPDDR5x (unified — CPU/GPU/NPU share) |
| OS | Ubuntu 26.04 LTS |
| Kernel | 7.0.0+ (in-tree `amdxdna` driver) |

## Results

| Benchmark | Result |
|---|---|
| 32-core BFP16 GEMM peak | **31.2 TFLOPS** |
| 32-core BFP16 GEMM sustained | **30.9 TFLOPS** |
| Gemma 4 2B (FastFlowLM) | **25 tok/s** |
| Llama 3.1 8B (FastFlowLM) | **15 tok/s** |

---

## Step 1 — Verify NPU

```bash
# NPU should show as RyzenAI-npu5
sudo xrt-smi examine
```

Expected output:
```
[0000:c6:00.1]  |RyzenAI-npu5  |
```

If not present, check the driver is loaded:
```bash
lsmod | grep amdxdna
sudo modprobe amdxdna
```

## Step 2 — Install XRT + Python bindings

```bash
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update
sudo apt install libxrt2 libxrt-npu2 libxrt-dev libxrt-utils \
                 libxrt-utils-npu python3-xrt
sudo usermod -aG render $USER
# log out and back in, or: newgrp render
```

## Step 3 — Set up IRON (Python NPU kernels, Apache 2.0)

```bash
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie
python3 -m venv .venv
source .venv/bin/activate
pip install numpy ml_dtypes

# Install prebuilt wheel
pip install https://github.com/Xilinx/mlir-aie/releases/download/v1.3.2/mlir_aie-1.3.2-cp314-cp314-manylinux_2_35_x86_64.whl
```

### Apply patches (v1.3.2 wheel is missing some exports)

```bash
cd .venv/lib/python3.14/site-packages/mlir_aie/python/aie/iron

# Add In/Out/CompileTime markers
cat >> __init__.py << 'PATCH'
class CompileTime: pass
class In: pass
class Out: pass
class InOut: pass
PATCH
```

## Step 4 — Install Peano (open-source AIE2P compiler, Apache 2.0)

```bash
pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-21.0.0.2026062501+c83e305a-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

export PEANO_INSTALL_DIR="$VIRTUAL_ENV/lib/python3.14/site-packages/llvm-aie"
```

## Step 5 — Verify with a passthrough test

```bash
cat > /tmp/pt.cc << 'EOF'
#include <stdint.h>
extern "C" void pt(int32_t* a, int32_t* c, int32_t n) {
    for (int i = 0; i < n; i++) c[i] = a[i];
}
EOF

source .venv/bin/activate
export PEANO_INSTALL_DIR="$VIRTUAL_ENV/lib/python3.14/site-packages/llvm-aie"

python3 -c "
import numpy as np, aie.iron as iron
from aie.iron import In, Out, ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

@iron.jit
def test(a_in: In, c_out: Out):
    ty = np.ndarray[(64,), np.dtype[np.int32]]
    k = ExternalFunction('pt', source_file='/tmp/pt.cc', arg_types=[ty,ty,np.int32])
    oi=ObjectFifo(ty,name='i'); oo=ObjectFifo(ty,name='o')
    def cf(i,o,k):
        ai=i.acquire(1); co=o.acquire(1); k(ai,co,64); o.release(1); i.release(1)
    w=Worker(cf,[oi.cons(),oo.prod(),k],tile=Tile(0,2))
    rt=Runtime()
    with rt.sequence(ty,ty) as (a,c): rt.start(w); rt.fill(oi.prod(),a); rt.drain(oo.cons(),c,wait=True)
    return Program(iron.get_current_device(),rt).resolve_program()

a=np.arange(64,dtype=np.int32); c=np.zeros(64,dtype=np.int32)
test(a,c)
print('PASS' if np.array_equal(c,a) else 'FAIL')
"
```

## Step 6 — Install Chess compiler (for 31 TFLOPS)

The Chess compiler requires a free license from AMD.

1. Register at **https://account.amd.com/en/member/ryzenai-sw-ea.html**
2. Download **Vitis AIE Essentials** (.whl)
3. Extract the license:
```bash
mkdir -p /tmp/vae && cd /tmp/vae
unzip ~/Downloads/vitis_aie_essentials*.whl
find . -name 'Xilinx.lic' -exec cp {} /home/bcloud/torch2aie/licenses/ \;
export XILINXD_LICENSE_FILE=/home/bcloud/torch2aie/licenses/Xilinx.lic
```

The license provides `AIEbuild`, `AIEMLbuild`, `AIEMLv2build` features — permanent, uncounted.

## Step 7 — Install torch2aie (prebuilt Chess + MLIR-AIE bundle)

```bash
git clone https://github.com/taowen/torch2aie.git
cd torch2aie
./scripts/install_toolchain_from_release.sh
./scripts/setup_python.sh
source ./scripts/env.sh
export XILINXD_LICENSE_FILE=/path/to/Xilinx.lic
```

## Step 8 — Run 31 TFLOPS GEMM

```bash
cd ~/torch2aie
source scripts/env.sh
export XILINXD_LICENSE_FILE=/path/to/Xilinx.lic
./scripts/run_atb_gemm.sh config2
```

Expected output:
```
Avg NPU tflops: 30.9
Max NPU tflops: 31.2
```

## Step 9 — Install FastFlowLM (LLM inference on NPU)

```bash
# Already available as a .deb:
sudo dpkg -i fastflowlm_0.9.43_ubuntu*.deb

# Or pull models:
/opt/fastflowlm/bin/flm pull gemma4-it:e2b
/opt/fastflowlm/bin/flm pull llama3.1:8b

# Run:
echo 'Hello!' | /opt/fastflowlm/bin/flm run gemma4-it:e2b
```

## License Notes

| Component | License | Shareable? |
|---|---|---|
| IRON (`Xilinx/mlir-aie`) | Apache 2.0 | ✅ Yes |
| Peano (`Xilinx/llvm-aie`) | Apache 2.0 w/ LLVM exceptions | ✅ Yes |
| Chess compiler | Requires per-user Xilinx.lic | ⚠️ License file is personal |
| Compiled `.xclbin` files | Derivative of Chess compiler | ⚠️ Share source, not binaries |
| FastFlowLM | AMD proprietary EULA | ⚠️ Check EULA |
| torch2aie toolchain | Contains Chess binaries | ⚠️ Not redistributable |

For open distribution, use **IRON + Peano** only — everything is Apache 2.0.

## References

- [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) — IRON API + MLIR-AIE toolchain
- [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie) — Peano (open-source AIE2P compiler)
- [taowen/torch2aie](https://github.com/taowen/torch2aie) — Prebuilt Chess + GEMM benchmarks
- [amd/RyzenAI-SW](https://github.com/amd/RyzenAI-SW) — AMD Ryzen AI Software
- [FastFlowLM](https://huggingface.co/FastFlowLM) — LLM runtime for Ryzen AI NPUs
- [Ryzen AI Docs](https://ryzenai.docs.amd.com/en/latest/) — Official AMD documentation

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

## Future-Frontend Bibliography

> **PyTorch Export IR** — The PyTorch 2.x `torch.export` intermediate representation (IR) specification (https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/export/ir_spec.html) defines a portable graph format that may serve as a front-end for NPU control-plane kernel workflows in future phases. This source is noted here for reference only and does not expand current MVP scope.

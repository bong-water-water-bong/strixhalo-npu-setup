# Clean-Room Chess Replacement Research Plan and Unified NPU Control Plane Design

Date: 2026-06-26
Status: Approved design, public clean-room research/specification
Repository: `strixhalo-npu-setup`

## 1. Goal

Build the public technical foundation for a full open-source replacement for the proprietary Chess compiler layer used in high-performance AMD/Xilinx AI Engine workflows, starting with AMD Ryzen AI Max+ 395 / Strix Halo AIE2P NPU.

The first deliverable is not a compiler implementation. It is a clean-room report and design that defines:

1. What the open stack already provides.
2. What Chess appears to provide at the performance-critical layer.
3. How to replicate Chess-level performance without copying proprietary internals.
4. Which experiments should be run first.
5. How a unified control plane should coordinate devices, toolchains, kernels, benchmarks, profiles, and future model dispatch.

## 2. Clean-Room Boundary

This project must be safe to publish as open-source work.

Allowed inputs:

- Public AMD/Xilinx documentation.
- Public source repositories such as `Xilinx/mlir-aie` and `Xilinx/llvm-aie`.
- Public AIE API and public intrinsic references.
- Public examples, public benchmark methodology, and independently authored experiments.
- Black-box performance numbers gathered by running legally installed tools, as long as proprietary generated artifacts or internals are not copied into the open project.

Disallowed inputs:

- Copying proprietary Chess compiler code, headers, binaries, or internal algorithms.
- Publishing private license files or license-derived proprietary artifacts.
- Decompiling proprietary compiler binaries.
- Copying generated proprietary code or binary layouts in a way that creates a derivative implementation.
- Depending on unpublished AMD/Xilinx internal documents.

Clean-room benchmark rule:

- It is acceptable to compare open kernels against Chess-produced performance numbers.
- It is acceptable to publish high-level metrics such as throughput, latency, utilization, instruction counts when legally exposed by public tools, and failure modes.
- It is not acceptable to publish or clone proprietary compiler internals.

## 3. Current Public/Open Stack

The Strix Halo NPU path can be decomposed as follows:

| Layer | Role | Public/Open Status |
|---|---|---|
| Linux `amdxdna` driver | Kernel driver for Ryzen AI NPU | In-tree / public |
| XRT | Runtime, buffers, execution, device management | Public packages |
| MLIR-AIE | AIE dialects, routing, lowering, packaging flows | Apache 2.0 |
| IRON | Python-level programming model for AIE kernels | Apache 2.0 via MLIR-AIE |
| Peano / LLVM-AIE | Open LLVM backend/toolchain for AIE targets | Apache 2.0 with LLVM exceptions |
| Chess | Proprietary AIE compiler/scheduler | Not open |
| torch2aie-style flows | High-performance reference workflows | Public repo, but may bundle or depend on non-redistributable tools |
| FastFlowLM-style runtime | Model execution on Ryzen AI | Proprietary/runtime-specific |

The open stack can already express kernels, allocate resources, route data movement, generate runtime artifacts, and compile simple AIE code. The missing high-performance layer is the part where Chess excels: producing tightly scheduled, low-spill, vectorized, bank-aware AIE machine code that fits program-memory limits and sustains peak compute.

## 4. What Chess Likely Contributes

From public behavior, documentation, compiler interfaces, and observable performance, Chess appears to provide several performance-critical functions.

### 4.1 VLIW Scheduling

AIE cores are VLIW-style processors with multiple issue slots and strict resource constraints. Peak performance requires packing scalar, vector, load/store, and control operations into bundles with minimal bubbles.

Open replacement requirement:

- Model AIE2P issue slots and hazards.
- Schedule vector MAC, load, store, shuffle, and scalar/control operations together.
- Minimize NOP density.
- Respect latency and forwarding rules.

### 4.2 Software Pipelining

High-performance kernels need loop bodies transformed so that loads, compute, and stores overlap.

Open replacement requirement:

- Generate or transform loops into pipelined forms.
- Support prologue/kernel/epilogue scheduling.
- Track initiation interval.
- Expose diagnostics when a loop cannot pipeline efficiently.

### 4.3 Register Allocation and Spill Avoidance

AIE kernels are register-sensitive. Spills to stack or local memory can destroy performance.

Open replacement requirement:

- Track vector, accumulator, scalar, and predicate register pressure.
- Prefer schedules and tile shapes that avoid spills.
- Provide clear spill reports.
- Allow explicit lifetime fences in generated code or MLIR.

### 4.4 Memory Bank Awareness

AIE tile memory is banked. Layouts that look equivalent at the algorithm level can produce very different performance depending on bank conflicts.

Open replacement requirement:

- Model L1 banks, DMA buffer placement, vector load alignment, and operand layout.
- Generate layouts for GEMM and attention that avoid dual-read conflicts.
- Validate layout choices with benchmarks.

### 4.5 Intrinsic Lowering

Public AIE APIs and intrinsics expose operations such as vector MACs, shuffles, conversions, and BFP16/BFP-style operations. The replacement does not need to invent all operations from scratch, but it must lower high-level operations into good intrinsic sequences.

Open replacement requirement:

- Prefer public AIE API / MLIR-AIE / AIEVec operations where available.
- Build a public intrinsic-pattern library for AIE2P.
- Keep generated code understandable and benchmarkable.

### 4.6 Program-Memory Fitting

AIE tile program memory is limited. Fully unrolled high-performance kernels can exceed limits.

Open replacement requirement:

- Track generated code size early.
- Choose unroll factors and specialization levels that fit.
- Provide code-size diagnostics per tile.

## 5. Open-Source Chess Replacement Architecture

The long-term replacement should be a layered compiler, not a monolithic clone.

```text
User kernel description
    ↓
Frontend
    - MLIR dialect input, Python DSL, or restricted C++/kernel template input
    ↓
Middle end
    - shape specialization
    - tiling
    - vectorization
    - memory layout
    - ObjectFIFO/DMA planning
    - bank assignment
    ↓
Open backend path v1
    - MLIR-AIE + AIEVec + Peano/LLVM-AIE
    ↓
Open backend path v2
    - custom clean-room AIE2P scheduler and diagnostics
    ↓
Artifacts
    - ELF/xclbin/instruction streams usable through XRT/amdxdna
```

### 5.1 Frontend Strategy

Do not begin with a general C/C++ compiler. Start with a narrow, inspectable kernel description format for performance-critical primitives.

Initial frontend options:

1. MLIR-first kernel specs.
2. Python DSL that emits MLIR-AIE / AIEVec.
3. Template-based kernel generator for GEMM and vector ops.

Recommendation: start with a template/DSL hybrid for GEMM and vector kernels, while keeping the internal representation MLIR-friendly.

### 5.2 Middle-End Strategy

The middle end should be responsible for architecture-aware choices:

- Tile shape selection.
- BFP16/BF16/FP32 datatype paths.
- Local memory layout.
- Double buffering / ping-pong buffering.
- ObjectFIFO depth.
- DMA block descriptor planning.
- Multi-tile partitioning.
- Accumulator layout and reduction strategy.

This is where much of the recoverable performance likely lives. It is also the safest area for open innovation because it can be derived from public architecture constraints and benchmark feedback.

### 5.3 Backend Strategy

Backend v1 should use Peano/LLVM-AIE as much as possible.

Backend v2 should focus narrowly on the missing scheduler:

- AIE2P machine model.
- Bundle formation.
- Hazard checks.
- Register-pressure-aware scheduling.
- Loop pipelining.
- Spill and NOP diagnostics.

The custom scheduler should be built only after the benchmark suite proves where Peano/LLVM-AIE falls short.

## 6. Performance Replication Plan

### 6.1 Core Metric

Primary metric:

- Sustained TFLOPS on BFP16/BF16 GEMM across 32 AIE2P tiles.

Secondary metrics:

- Single-tile GOPS/TFLOPS.
- Multi-tile scaling efficiency.
- DMA stall time.
- NOP density or equivalent scheduling waste.
- Register spills.
- Program-memory usage.
- Numerical error versus CPU reference.

### 6.2 Experiment 1 — Single-Tile Kernel Baseline

Goal: establish the performance ceiling for one AIE2P tile using only open tooling.

Tasks:

1. Implement passthrough and vector add to validate data movement.
2. Implement a small matmul microkernel.
3. Add BFP16/BF16 path if public intrinsics support it cleanly.
4. Measure warmup-adjusted latency and throughput.
5. Record generated artifact size and any compile diagnostics.

Pass criteria:

- Correctness against CPU reference.
- Repeatable timings.
- Clear bottleneck attribution: compute, memory, DMA, scheduling, or host overhead.

### 6.3 Experiment 2 — Memory Layout and Bank Conflicts

Goal: quantify how much layout matters.

Tasks:

1. Benchmark naive row-major layout.
2. Benchmark blocked 8x8 layout.
3. Benchmark asymmetric tile buffering-style layouts.
4. Sweep alignment, strides, and buffer placement.
5. Compare DMA time, compute time, and total time.

Pass criteria:

- Identify at least one layout that outperforms naive layout.
- Produce a public layout guide for GEMM operands.

### 6.4 Experiment 3 — Software Pipelining and Double Buffering

Goal: overlap data movement and compute.

Tasks:

1. Single-buffer baseline.
2. Ping-pong local buffers.
3. ObjectFIFO depth sweep.
4. DMA BD chain variants.
5. Compare tile utilization.

Pass criteria:

- Demonstrate measurable improvement from overlap.
- Document the minimum buffering pattern needed for sustained throughput.

### 6.5 Experiment 4 — Multi-Tile Scaling

Goal: scale from one tile to 32 tiles.

Tasks:

1. 1-tile baseline.
2. 2x2 cluster.
3. 4x4 cluster.
4. 8x4 or full 32-tile mapping, depending on Strix Halo topology.
5. Evaluate strong and weak scaling.

Pass criteria:

- Scaling curve with efficiency numbers.
- Identify communication bottlenecks.
- Produce a public mapping strategy for full-NPU GEMM.

### 6.6 Experiment 5 — Accuracy and Datatype Tradeoffs

Goal: characterize BFP16/BF16/FP32 behavior.

Tasks:

1. Compare FP32, BF16, and BFP16-style accumulation where supported.
2. Test random matrices, structured matrices, and model-like distributions.
3. Track max error, mean error, and relative error.
4. Measure throughput per datatype.

Pass criteria:

- Public accuracy/performance curve.
- Recommended datatype path for LLM GEMM.

### 6.7 Experiment 6 — Attention Primitive Follow-Up

After GEMM is understood, extend to attention primitives:

- QK^T.
- Softmax or approximate softmax.
- PV.
- Fused attention blocks where feasible.

This should not block the GEMM-first compiler path.

## 7. Research Report Deliverables

The public report should include:

1. A clean-room policy.
2. A public-source bibliography.
3. Stack diagram.
4. Gap analysis of Chess versus open tooling.
5. Benchmark methodology.
6. Experiment matrix.
7. Compiler architecture proposal.
8. Unified control plane proposal.
9. Roadmap and milestones.

The report should avoid claiming exact Chess internals. It should use language such as "likely contributes", "publicly observable", and "hypothesis to benchmark" unless a fact is directly supported by public documentation or source code.

## 8. Public Source Bibliography

Primary public sources to use for the research and implementation phases:

| Source | Public location | Use |
|---|---|---|
| MLIR-AIE | `https://github.com/Xilinx/mlir-aie` | AIE dialects, IRON, examples, ObjectFIFO patterns, routing/lowering flows |
| LLVM-AIE / Peano | `https://github.com/Xilinx/llvm-aie` | Open AIE compiler backend and LLVM toolchain path |
| AIE API documentation | `https://xilinx.github.io/aie_api/` and AMD docnav AIE API pages | Public vector API semantics and portable programming model |
| AIE-ML / AIE2 public intrinsic docs | AMD public documentation portal / docnav | Public intrinsic behavior and datatype references |
| Ryzen AI software documentation | `https://ryzenai.docs.amd.com/` | Runtime stack, supported platforms, XRT integration guidance |
| XRT documentation/source | `https://github.com/Xilinx/XRT` and package docs | Runtime execution, buffer management, profiling hooks |
| Linux `amdxdna` driver | Linux kernel source and distro packages | Device/driver interface context |
| Public torch2aie repository | `https://github.com/taowen/torch2aie` | Public benchmark context and integration patterns; do not copy proprietary bundled artifacts |

Use these sources for public facts, APIs, and benchmark methodology. If a claim cannot be traced to a public source or to an independently generated experiment, mark it as a hypothesis.

## 9. Unified Control Plane Side Quest

A compiler project needs a control plane so experiments do not become scattered shell scripts and temporary files.

The control plane should coordinate:

- Device discovery.
- Toolchain discovery.
- Kernel compilation.
- Artifact caching.
- Benchmark execution.
- Profile collection.
- Metadata persistence.
- Future model-dispatch planning.

### 9.1 Proposed Command Surface

```bash
npu-ctrl discover
npu-ctrl status
npu-ctrl toolchain probe
npu-ctrl kernels list
npu-ctrl kernels compile <kernel>
npu-ctrl kernels purge
npu-ctrl bench run <kernel> --iters 100 --warmup 5
npu-ctrl bench sweep --kernel gemm --sizes 64,128,256,512
npu-ctrl bench report
npu-ctrl profile collect <kernel>
npu-ctrl profile report
npu-ctrl dispatch plan <model>
```

### 9.2 Proposed Modules

```text
control_plane/
  discovery          # NPU/XRT/driver/device probing
  toolchains         # Peano, IRON, Chess availability reporting
  kernel_registry    # artifact cache and metadata
  builder            # compile jobs and source hashing
  benchmark_driver   # repeatable benchmark execution
  profile_collector  # XRT/profile data collection
  resource_manager   # tile/memory allocation model
  model_dispatcher   # future model graph routing
  metadata_store     # JSON/SQLite-backed state
  cli                # npu-ctrl command surface
```

### 9.3 Metadata Store

Default location:

```text
~/.npu_control_plane/store/
```

Suggested layout:

```text
devices.json
toolchains.json
registry/
  kernels/
    index.json
    artifacts/
benchmarks/
  runs/
  summary.json
profiles/
models/
```

Example device record:

```json
{
  "id": 0,
  "bdf": "0000:c6:00.1",
  "name": "RyzenAI-npu5",
  "tile_count": 32,
  "peak_tflops_claimed": 31.2,
  "driver": "amdxdna",
  "runtime": "xrt",
  "last_seen": "2026-06-26T00:00:00Z"
}
```

Example toolchain record:

```json
{
  "toolchains": [
    {
      "name": "peano",
      "available": true,
      "path": "$PEANO_INSTALL_DIR",
      "license_required": false
    },
    {
      "name": "iron",
      "available": true,
      "path": "python environment containing aie.iron",
      "license_required": false
    },
    {
      "name": "chess",
      "available": false,
      "license_required": true,
      "public_clean_room_use": "benchmark reference only; do not redistribute artifacts"
    }
  ]
}
```

### 9.4 Control Plane MVP

MVP scope:

1. `npu-ctrl discover`.
2. `npu-ctrl status`.
3. `npu-ctrl toolchain probe`.
4. `npu-ctrl kernels list`.
5. JSON metadata store.
6. Kernel artifact registry keyed by source hash, target, shape, dtype, and toolchain.

Deferred:

- Model graph dispatch.
- Advanced profiling UI.
- Multi-kernel resource scheduling.
- Auto-tuning.
- Full compiler integration.

## 10. Roadmap

### Phase 0 — Public Report and Spec

Deliver this clean-room report and control-plane design.

Exit criteria:

- Spec is committed.
- Clean-room boundaries are explicit.
- Experiment roadmap is actionable.

### Phase 1 — Measurement Harness

Build repeatable benchmark infrastructure.

Deliverables:

- `npu-ctrl` MVP or equivalent scripts.
- Kernel registry.
- Benchmark result JSON schema.
- Baseline passthrough/vector/GEMM measurements.

### Phase 2 — GEMM-First Open Compiler Path

Build a narrow compiler/generator for GEMM.

Deliverables:

- Shape-specialized GEMM generator.
- Public tiling/layout library.
- Single-tile and multi-tile benchmarks.
- Correctness tests.

### Phase 3 — Scheduler Gap Analysis

Determine whether Peano/LLVM-AIE is sufficient or whether a custom clean-room AIE2P scheduler is required.

Deliverables:

- Schedule diagnostics.
- Spill reports.
- Code-size reports.
- NOP/idle-cycle analysis where publicly measurable.

### Phase 4 — Clean-Room AIE2P Scheduler Prototype

Build a narrow scheduler for known kernel patterns.

Deliverables:

- Machine model.
- Bundle verifier.
- Hazard checker.
- GEMM microkernel scheduler.
- Comparison against Peano-generated code.

### Phase 5 — Expanded Compiler

Extend beyond GEMM.

Candidate primitives:

- GEMV.
- Batched GEMM.
- QK^T.
- Softmax.
- PV.
- Quantized matmul.
- Fused LLM blocks.

## 11. Key Risks

| Risk | Severity | Mitigation |
|---|---:|---|
| Proprietary boundary mistakes | High | Maintain explicit clean-room policy and avoid generated proprietary artifacts |
| AIE2P scheduling complexity | High | Start with narrow GEMM scheduler, not general compiler |
| Peano performance gap | Medium/High | Benchmark first; customize only proven bottlenecks |
| Insufficient public documentation | Medium | Use empirical black-box benchmark methodology without copying internals |
| Program-memory overflows | Medium | Add code-size diagnostics early |
| Bank conflicts | Medium | Make layout experiments first-class |
| Runtime fragmentation | Medium | Build unified control plane metadata and CLI early |

## 12. Success Criteria

Short-term success:

- Public report explains the path to a Chess replacement without violating clean-room constraints.
- Benchmark plan can be executed by another developer.
- Control plane MVP scope is clear.

Medium-term success:

- Open GEMM kernels demonstrate repeatable, improving performance on Strix Halo.
- Artifact registry and benchmark data make regressions visible.
- Peano versus custom-scheduler bottlenecks are quantified.

Long-term success:

- A public compiler path generates high-performance AIE2P kernels without requiring Chess.
- GEMM and attention primitives are fast enough to support practical LLM workloads on Ryzen AI NPU.
- The project provides transparent diagnostics: scheduling quality, memory layout, spills, DMA stalls, and code size.

## 13. Immediate Next Steps

1. Commit this spec.
2. Create an implementation plan for Phase 1: measurement harness and `npu-ctrl` MVP.
3. Build the control-plane skeleton.
4. Add repeatable benchmark schemas.
5. Run first open-stack baseline kernels.

Implementation should not start until the written spec has been reviewed and approved.

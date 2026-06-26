# NPU Benchmark Results

## Passthrough DMA Baseline (IRON + Peano, open-source)

| N (bytes) | Avg (µs) | Min (µs) | Max (µs) | Correct |
|-----------|----------|----------|----------|---------|
| 64        | 138.0    | 118.8    | 163.7    | ✅      |
| 256       | 139.0    | 114.4    | 238.0    | ✅      |
| 1024      | 217.1    | 121.6    | 353.5    | ✅      |
| 4096      | 123.6    | 115.7    | 136.8    | ✅      |
| 16384     | 177.4    | 139.8    | 240.9    | ✅      |

**Kernel**: shim → memtile → shim DMA forward (no compute tile).
**Toolchain**: IRON (mlir-aie) + Peano, Apache 2.0.

## Chess GEMM Reference (torch2aie, license-required)

| Metric | Value |
|--------|-------|
| Matrix | 3072×4096×1536 |
| Avg TFLOPS | 30.1 |
| Avg NPU time | 1283 µs |
| Tiles | 32 AIE2P cores (full NPU) |

**Toolchain**: Chess compiler via Xilinx.lic (free EA license).
**Status**: Benchmark reference only, not redistributable.

## Gap

The open-source passthrough establishes data movement costs (~120-220 µs per transaction). The Chess GEMM achieves 30.1 TFLOPS on compute. Closing this gap requires:
1. Single-tile GEMM microkernel via IRON + Peano
2. Multi-tile scaling
3. BFP16 vector MAC intrinsics
4. Custom clean-room AIE2P scheduler (if Peano gap exceeds ~2×)

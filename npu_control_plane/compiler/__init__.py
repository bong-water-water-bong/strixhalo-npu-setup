"""Clean-room AIE2P compiler/scheduler for NPU GEMM optimization.

Implements a complete VLIW modulo scheduler for the Strix Halo NPU (AIE2P)
architecture, derived from public documentation and the open-source mlir-aie
and llvm-aie (Peano) projects. No Chess/proprietary internals are referenced.

Modules:
- machine_model:  AIE2P VLIW architecture model (7-way, 64KB L1, BFP16 intrinsics)
- scheduler:      Swing Modulo Scheduling (SMS) for software-pipelined GEMM loops
- bank_allocator: GAMA-style bank-conflict-aware L1 buffer allocation
- gemm_optimizer: GEMM tile size optimization and performance analysis
- kernel_generator: C++ and IRON Python kernel code generator

Key architectural findings (from public sources):
- AIE2P L1: 64 KB (4 banks × 16 KB), not 32 KB
- VLIW: 7-way (scalar + 2 moves + 2 loads + 1 store + 1 vector)
- BFP16: MacConfBFP576ACC2048 intrinsic → 512 MACs/insn (2× vs BF16)
- II=1 achievable with 4-accumulator interleaving
- Peano doesn't auto-software-pipeline (Issue #126)
- Bank conflicts: 1 cycle penalty, fixable with padding + GAMA layout
"""

from .machine_model import (
    AIE2PMachineModel,
    AIE2PVLIWBundle,
    RegisterClass,
    RegisterPressure,
    InstructionLatency,
    BFP16Config,
    BF16Config,
    INT8Config,
    GemmMicroKernel,
    IssueSlot,
    AIE2P_CLOCK_MHZ,
    AIE2P_L1_DATA_BYTES,
    AIE2P_L1_BANKS,
    AIE2P_VECTOR_REGS,
    AIE2P_ACCUMULATOR_REGS,
)
from .scheduler import (
    VLIWScheduler,
    ScheduledLoop,
    gemm_inner_loop_instructions,
    gemm_store_instructions,
    Instruction,
    OpType,
    DepGraphBuilder,
    DepNode,
    DepEdge,
)
from .bank_allocator import (
    BankAllocator,
    BankLayout,
    BufferPlacement,
    BufferClass,
)
from .gemm_optimizer import (
    GemmLoopOptimizer,
    GemmProblem,
    TileConfig,
)
from .kernel_generator import (
    KernelCodeGenerator,
    KernelConfig,
)

__all__ = [
    # Machine model
    "AIE2PMachineModel",
    "AIE2PVLIWBundle",
    "RegisterClass",
    "RegisterPressure",
    "InstructionLatency",
    "BFP16Config",
    "BF16Config",
    "INT8Config",
    "GemmMicroKernel",
    "IssueSlot",
    "AIE2P_CLOCK_MHZ",
    "AIE2P_L1_DATA_BYTES",
    "AIE2P_L1_BANKS",
    "AIE2P_VECTOR_REGS",
    "AIE2P_ACCUMULATOR_REGS",
    # Scheduler
    "VLIWScheduler",
    "ScheduledLoop",
    "gemm_inner_loop_instructions",
    "gemm_store_instructions",
    "Instruction",
    "OpType",
    "DepGraphBuilder",
    "DepNode",
    "DepEdge",
    # Bank allocator
    "BankAllocator",
    "BankLayout",
    "BufferPlacement",
    "BufferClass",
    # GEMM optimizer
    "GemmLoopOptimizer",
    "GemmProblem",
    "TileConfig",
    # Kernel generator
    "KernelCodeGenerator",
    "KernelConfig",
]

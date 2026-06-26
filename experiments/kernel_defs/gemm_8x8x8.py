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
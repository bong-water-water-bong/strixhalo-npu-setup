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
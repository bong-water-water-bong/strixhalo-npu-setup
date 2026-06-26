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
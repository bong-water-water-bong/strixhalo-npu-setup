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
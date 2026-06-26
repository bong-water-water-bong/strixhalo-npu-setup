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
    c_dtype: DataType = field(default=None)
    layout: Layout = field(default_factory=RowMajor)
    multi_tile: MultiTileConfig = field(default_factory=lambda: MultiTileConfig(2, 2))

    def __post_init__(self):
        if self.c_dtype is None:
            object.__setattr__(self, "c_dtype", BFP16)
        self._tile = TileShape(self.block_m, self.block_k, self.block_n)

    def fits_in_l1(self, dtype: DataType = None) -> bool:
        return self._tile.fits_in_l1(dtype=dtype or self.a_dtype)

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
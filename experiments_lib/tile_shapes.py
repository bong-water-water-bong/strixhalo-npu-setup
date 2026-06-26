from dataclasses import dataclass, field

from .datatypes import DataType, FP32, BFP16

AIE2P_L1_BYTES = 32 * 1024


@dataclass(frozen=True)
class TileShape:
    m: int
    k: int
    n: int

    def byte_count(self, dtype: DataType = BFP16) -> int:
        """Total L1 bytes needed for A+B+C buffers."""
        a_bytes = self.m * self.k * dtype.element_size
        b_bytes = self.k * self.n * dtype.element_size
        c_bytes = self.m * self.n * FP32.element_size  # accumulate in FP32
        return a_bytes + b_bytes + c_bytes

    def fits_in_l1(self, cfg: "TileShapeConfig" = None, dtype: DataType = BFP16) -> bool:
        return self.byte_count(dtype) <= AIE2P_L1_BYTES


@dataclass(frozen=True)
class TileShapeConfig:
    m: int = 64
    k: int = 64
    n: int = 64

    @property
    def base(self) -> TileShape:
        return TileShape(self.m, self.k, self.n)

    def plausible_shapes(self) -> list[TileShape]:
        return [
            TileShape(32, 32, 32),
            TileShape(32, 64, 64),
            TileShape(64, 64, 64),
        ]


@dataclass
class MultiTileConfig:
    rows: int = 2
    cols: int = 2

    @property
    def tile_count(self) -> int:
        return self.rows * self.cols
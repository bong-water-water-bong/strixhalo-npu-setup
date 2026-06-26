from dataclasses import dataclass
from typing import Any


class Layout:
    name: str


@dataclass(frozen=True)
class RowMajor(Layout):
    name: str = "row-major"
    description: str = "naive row-major layout, default"


@dataclass(frozen=True)
class BlockedLayout(Layout):
    block_size: int = 8
    name: str = "blocked-8x8"

    def __post_init__(self):
        object.__setattr__(self, "name", f"blocked-{self.block_size}x{self.block_size}")

    @property
    def description(self):
        return f"{self.block_size}x{self.block_size} blocked layout"


@dataclass(frozen=True)
class ATBLayout(Layout):
    a_super: int = 2
    b_super: int = 1
    name: str = "atb-2x1"

    def __post_init__(self):
        object.__setattr__(self, "name", f"atb-{self.a_super}x{self.b_super}")

    @property
    def description(self):
        return f"Asymmetric tile buffering {self.a_super}x{self.b_super}"
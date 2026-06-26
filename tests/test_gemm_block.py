"""Tests for GemmBlockKernel (Task 3)."""

from experiments.kernel_defs.gemm_block import GemmBlockKernel
from experiments_lib.layouts import RowMajor, BlockedLayout, ATBLayout


def test_gemm_block_descriptor():
    """GemmBlockKernel has correct block dimensions."""
    k = GemmBlockKernel(name="gblock", block_m=64, block_k=64, block_n=64)
    assert k.block_m == 64
    assert k.block_k == 64
    assert k.block_n == 64


def test_gemm_block_tile_count():
    """GemmBlockKernel reports correct tile count from MultiTileConfig."""
    k = GemmBlockKernel(name="gblock", block_m=64, block_k=64, block_n=64)
    assert k.tile_count() == 4  # 2x2 default


def test_gemm_block_layout_string():
    """GemmBlockKernel layout string includes layout name."""
    k = GemmBlockKernel(name="gblock", layout=ATBLayout(2, 1))
    assert "atb" in str(k.layout).lower()


def test_gemm_block_str():
    """GemmBlockKernel has a string representation."""
    k = GemmBlockKernel(name="gblock", block_m=64, block_k=64, block_n=64)
    s = str(k)
    assert "gblock" in s
    assert "64" in s
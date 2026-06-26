"""Tests for experiment infrastructure. These do NOT require NPU hardware."""

from experiments.runner import NpuRunner, ExperimentConfig
from experiments.kernel_defs.passthrough import PassthroughKernel
from experiments_lib.layouts import RowMajor, BlockedLayout, ATBLayout
from experiments_lib.datatypes import BFP16, BF16, FP32
from experiments_lib.tile_shapes import TileShape, TileShapeConfig

# NEW: import for Task 2 kernel descriptors
from experiments.kernel_defs.vec_add import VecAddKernel
from experiments.kernel_defs.gemm_8x8x8 import Gemm8x8x8Kernel


def test_passthrough_kernel_descriptor():
    """Passthrough kernel descriptor has expected metadata without hardware."""
    k = PassthroughKernel(name="pt")
    assert k.name == "pt"
    assert k.dtype == FP32
    assert k.tile_shape is None  # no tiling for passthrough
    assert "passthrough" in str(k).lower()


def test_npu_runner_config():
    """ExperimentConfig stores metadata correctly."""
    cfg = ExperimentConfig(
        label="pt-test",
        kernel_name="passthrough",
        shape_str="N=1024",
        warmup=3,
        iters=20,
        toolchain="iron+peano",
        tile_ids=[0, 2],
    )
    assert cfg.label == "pt-test"
    assert cfg.warmup == 3
    assert cfg.iters == 20


def test_layout_strings():
    """Layout descriptors produce canonical strings."""
    assert RowMajor().name == "row-major"
    assert BlockedLayout(block_size=8).name == "blocked-8x8"
    assert ATBLayout(a_super=2, b_super=1).name == "atb-2x1"


def test_datatype_properties():
    """Datatype descriptors report correct widths."""
    assert BFP16.bit_width == 16
    assert BF16.bit_width == 16
    assert FP32.bit_width == 32
    assert BFP16.element_size == 2
    assert FP32.element_size == 4


def test_tile_shape_estimates():
    """Tile shape L1 fit estimation."""
    cfg = TileShapeConfig(m=64, k=64, n=64)
    shape = TileShape(32, 32, 32)
    # 32x32x32 BFP16: A=32*32*2=2048, B=2048, C=32*32*4=4096 = 8192 bytes
    assert shape.byte_count(dtype=BFP16) == 32 * 32 * 2 * 2 + 32 * 32 * 4
    assert shape.fits_in_l1(cfg, dtype=BFP16) is True


# =============================================================================
# Task 2: VecAddKernel and Gemm8x8x8Kernel
# =============================================================================


def test_vec_add_kernel_descriptor():
    k = VecAddKernel(name="vadd")
    assert k.name == "vadd"
    source = k.generate_iron_source(n=1024)
    assert "N = 1024" in source
    assert "for" in source.lower() or "range" in source


def test_gemm_8x8x8_kernel_descriptor():
    k = Gemm8x8x8Kernel(name="gemm8")
    assert k.name == "gemm8"
    source = k.generate_iron_source()
    assert "matmul" in source.lower() or "mmul" in source.lower()

# =============================================================================
# Task 4: Report generation
# =============================================================================


def test_generate_markdown_summary_empty(tmp_path):
    from experiments_lib.report import generate_markdown_summary
    from npu_control_plane.metadata import MetadataStore
    store = MetadataStore(tmp_path / "store")
    md = generate_markdown_summary(store)
    assert "Benchmark Summary" in md
    assert "No runs recorded" in md


def test_generate_markdown_summary_with_runs(monkeypatch, tmp_path):
    from experiments_lib.report import generate_markdown_summary
    from npu_control_plane.metadata import MetadataStore
    store = MetadataStore(tmp_path / "store")
    store.write_json("benchmarks", "summary.json", data={
        "runs": [{"label": "exp1", "median_ms": 12.3, "timestamp": "2026-01-01", "returncode": 0}]
    })
    md = generate_markdown_summary(store)
    assert "exp1" in md
    assert "12.300" in md


def test_generate_html_summary(monkeypatch, tmp_path):
    from experiments_lib.report import generate_html_summary
    from npu_control_plane.metadata import MetadataStore
    store = MetadataStore(tmp_path / "store")
    html = generate_html_summary(store)
    assert "<html>" in html
    assert "No runs recorded" in html

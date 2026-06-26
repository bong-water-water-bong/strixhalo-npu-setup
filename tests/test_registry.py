import json

from npu_control_plane.metadata import MetadataStore
from npu_control_plane.registry import KernelRegistry


def test_registry_starts_empty(tmp_path):
    registry = KernelRegistry(MetadataStore(tmp_path / "store"))

    assert registry.list() == []


def test_registry_registers_artifact_with_content_hash(tmp_path):
    artifact = tmp_path / "kernel.elf"
    artifact.write_bytes(b"public-test-artifact")
    registry = KernelRegistry(MetadataStore(tmp_path / "store"))

    record = registry.register(
        name="vec_add",
        artifact=artifact,
        dtype="i32",
        shape="N=64",
        toolchain="peano",
    )

    assert record["key"].startswith("vec_add__N=64__i32__peano__")
    assert record["name"] == "vec_add"
    assert record["artifact"].startswith("registry/kernels/artifacts/")
    copied = tmp_path / "store" / record["artifact"]
    assert copied.read_bytes() == b"public-test-artifact"
    index = json.loads((tmp_path / "store" / "registry" / "kernels" / "index.json").read_text())
    assert index["kernels"] == [record]


def test_registry_replaces_same_key_without_duplicates(tmp_path):
    artifact = tmp_path / "kernel.elf"
    artifact.write_bytes(b"public-test-artifact")
    registry = KernelRegistry(MetadataStore(tmp_path / "store"))

    first = registry.register("vec_add", artifact, "i32", "N=64", "peano", source_hash="abc")
    second = registry.register("vec_add", artifact, "i32", "N=64", "peano", source_hash="abc")

    assert first["key"] == second["key"]
    assert registry.list() == [second]


from npu_control_plane.cli import main


def test_cli_kernels_list_uses_store_env(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

    code = main(["kernels", "list"])

    captured = capsys.readouterr()
    assert code == 0
    assert '"kernels": []' in captured.out
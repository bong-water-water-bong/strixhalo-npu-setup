import json
from pathlib import Path

from npu_control_plane.metadata import MetadataStore, default_store_dir


def test_default_store_dir_uses_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "custom-store"))

    assert default_store_dir() == tmp_path / "custom-store"


def test_metadata_store_writes_pretty_json(tmp_path):
    store = MetadataStore(tmp_path / "store")

    written = store.write_json("devices.json", data={"devices": [{"name": "RyzenAI-npu5"}]})

    assert written == tmp_path / "store" / "devices.json"
    assert json.loads(written.read_text()) == {"devices": [{"name": "RyzenAI-npu5"}]}
    assert written.read_text().endswith("\n")
    assert "  \"devices\"" in written.read_text()


def test_metadata_store_read_json_returns_default_for_missing_file(tmp_path):
    store = MetadataStore(tmp_path / "store")

    assert store.read_json("missing.json", default={"ok": False}) == {"ok": False}


def test_metadata_store_rejects_paths_that_escape_root(tmp_path):
    store = MetadataStore(tmp_path / "store")

    try:
        store.path("..", "outside.json")
    except ValueError as exc:
        assert "escape" in str(exc)
    else:
        raise AssertionError("expected ValueError")

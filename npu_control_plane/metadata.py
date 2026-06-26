from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ENV_STORE = "NPU_CTRL_STORE"


def default_store_dir() -> Path:
    """Return the metadata store directory.

    Tests and reproducible runs can override the default with NPU_CTRL_STORE.
    """

    override = os.environ.get(ENV_STORE)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".npu_control_plane" / "store"


class MetadataStore:
    """Small JSON metadata store rooted at one directory."""

    def __init__(self, root: Path | None = None):
        self.root = (root or default_store_dir()).expanduser().resolve()

    def path(self, *parts: str) -> Path:
        candidate = self.root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"metadata path would escape store root: {candidate}") from exc
        return candidate

    def read_json(self, *parts: str, default: Any = None) -> Any:
        path = self.path(*parts)
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_json(self, *parts: str, data: Any) -> Path:
        path = self.path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
        return path

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from .metadata import MetadataStore


class KernelRegistry:
    def __init__(self, store: MetadataStore | None = None):
        self.store = store or MetadataStore()

    def _index(self) -> dict[str, Any]:
        return self.store.read_json("registry", "kernels", "index.json", default={"kernels": []})

    def _write_index(self, index: dict[str, Any]) -> None:
        self.store.write_json("registry", "kernels", "index.json", data=index)

    def list(self) -> list[dict[str, Any]]:
        return list(self._index().get("kernels", []))

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def register(
        self,
        name: str,
        artifact: Path,
        dtype: str,
        shape: str,
        toolchain: str,
        source_hash: str | None = None,
    ) -> dict[str, Any]:
        artifact = artifact.expanduser().resolve()
        if not artifact.is_file():
            raise FileNotFoundError(str(artifact))
        artifact_hash = self._hash_file(artifact)
        source = source_hash or artifact_hash
        key = f"{name}__{shape}__{dtype}__{toolchain}__{source[:12]}"
        suffix = artifact.suffix or ".bin"
        relative_artifact = f"registry/kernels/artifacts/{key}{suffix}"
        destination = self.store.path(*relative_artifact.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, destination)
        record = {
            "key": key,
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "toolchain": toolchain,
            "source_hash": source,
            "artifact_hash": artifact_hash,
            "artifact": relative_artifact,
        }
        index = self._index()
        kernels = [item for item in index.get("kernels", []) if item.get("key") != key]
        kernels.append(record)
        index["kernels"] = sorted(kernels, key=lambda item: item["key"])
        self._write_index(index)
        return record
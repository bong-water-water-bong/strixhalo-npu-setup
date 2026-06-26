from __future__ import annotations

import uuid
from datetime import datetime, timezone
from statistics import median
from time import perf_counter
from typing import Any

from .metadata import MetadataStore
from .probe import run_command


class BenchmarkRecorder:
    def __init__(self, store: MetadataStore | None = None):
        self.store = store or MetadataStore()

    def list_runs(self) -> list[dict[str, Any]]:
        summary = self.store.read_json("benchmarks", "summary.json", default={"runs": []})
        return list(summary.get("runs", []))

    def _write_summary(self, runs: list[dict[str, Any]]) -> None:
        self.store.write_json("benchmarks", "summary.json", data={"runs": runs})

    def record_command(self, label: str, command: list[str], warmup: int, iters: int) -> dict[str, Any]:
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        if iters <= 0:
            raise ValueError("iters must be > 0")
        durations_ms: list[float] = []
        last_result = None
        for index in range(warmup + iters):
            start = perf_counter()
            last_result = run_command(command, timeout=300)
            elapsed_ms = (perf_counter() - start) * 1000.0
            if index >= warmup:
                durations_ms.append(elapsed_ms)
            if last_result.returncode != 0:
                break
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        median_ms = median(durations_ms) if durations_ms else None
        returncode = last_result.returncode if last_result else None

        # Full record stored in per-run JSON file
        record = {
            "run_id": run_id,
            "label": label,
            "command": command,
            "timestamp": timestamp,
            "warmup": warmup,
            "iters": iters,
            "durations_ms": durations_ms,
            "median_ms": median_ms,
            "returncode": returncode,
            "stdout_tail": (last_result.stdout[-4000:] if last_result else ""),
            "stderr_tail": (last_result.stderr[-4000:] if last_result else ""),
        }
        self.store.write_json("benchmarks", "runs", f"{run_id}.json", data=record)

        # Lightweight summary entry for summary.json
        summary_entry = {
            "run_id": run_id,
            "label": label,
            "timestamp": timestamp,
            "median_ms": median_ms,
            "returncode": returncode,
            "warmup": warmup,
            "iters": iters,
            "command": command,
        }
        runs = [item for item in self.list_runs() if item.get("run_id") != run_id]
        runs.append(summary_entry)
        self._write_summary(runs)
        return record

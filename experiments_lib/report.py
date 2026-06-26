"""Report generation from npu-ctrl benchmark results."""

from __future__ import annotations
from typing import Any
from npu_control_plane.metadata import MetadataStore


def generate_markdown_summary(store: MetadataStore) -> str:
    """Generate a Markdown summary of all benchmark runs in store."""
    runs = store.read_json("benchmarks", "summary.json", default={"runs": []}).get("runs", [])
    if not runs:
        return "# Benchmark Summary\n\nNo runs recorded.\n"

    lines = ["# Benchmark Summary\n"]
    for run in runs:
        median = run.get("median_ms")
        label = run.get("label", "unknown")
        ts = run.get("timestamp", "")
        rc = run.get("returncode")
        lines.append(f"- **{label}** ({ts})")
        if median is not None:
            lines.append(f"  - Median: {median:.3f} ms")
        lines.append(f"  - Return code: {rc}")
        lines.append("")
    return "\n".join(lines)


def generate_html_summary(store: MetadataStore) -> str:
    """Generate a minimal HTML summary page."""
    md = generate_markdown_summary(store)
    return f"<html><body><pre>{md}</pre></body></html>"


__all__ = ["generate_markdown_summary", "generate_html_summary"]

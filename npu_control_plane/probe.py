from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def which(name: str) -> str | None:
    return shutil.which(name)


def run_command(args: Sequence[str], timeout: int = 5) -> CommandResult:
    try:
        completed = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(list(args), completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        return CommandResult(list(args), 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(list(args), 124, exc.stdout or "", exc.stderr or f"timed out after {timeout}s")
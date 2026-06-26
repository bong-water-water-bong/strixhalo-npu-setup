"""Tests for final review fixes: LICENSE, polite errors, consistent CLI exit codes."""

import sys
from pathlib import Path

import pytest

from npu_control_plane.cli import main


class TestLicenseFile:
    def test_root_license_file_exists(self):
        root = Path(__file__).resolve().parents[1]
        license_path = root / "LICENSE"
        assert license_path.exists(), "root LICENSE file should exist for Apache-2.0 project"
        text = license_path.read_text()
        assert "Apache License" in text
        assert "Version 2.0" in text


class TestKernelsRegisterMissingArtifact:
    def test_register_missing_artifact_returns_nonzero(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

        code = main(["kernels", "register", "--name", "test", "--artifact", "/nonexistent.elf", "--dtype", "i32", "--shape", "N=64", "--toolchain", "peano"])

        assert code != 0
        captured = capsys.readouterr()
        assert "nonexistent.elf" in captured.err or "not found" in captured.err.lower()
        # Should NOT contain a Python traceback
        assert "Traceback" not in captured.err
        assert "FileNotFoundError" not in captured.err

    def test_register_missing_artifact_no_traceback_contains_message(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

        code = main(["kernels", "register", "--name", "test", "--artifact", "/nonexistent.elf", "--dtype", "i32", "--shape", "N=64", "--toolchain", "peano"])

        captured = capsys.readouterr()
        assert captured.err.strip() != "", "stderr should contain a clear error message"
        # No Python exception formatting
        assert "Traceback" not in captured.err


class TestMissingSubcommandErrors:
    def test_toolchain_missing_subcommand_returns_2(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

        code = main(["toolchain"])

        assert code == 2
        captured = capsys.readouterr()
        assert "toolchain" in captured.err
        assert "subcommand" in captured.err.lower()
        # Should NOT say "command not implemented yet"
        assert "command not implemented yet" not in captured.err

    def test_kernels_missing_subcommand_returns_2(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

        code = main(["kernels"])

        assert code == 2
        captured = capsys.readouterr()
        assert "kernels" in captured.err
        assert "subcommand" in captured.err.lower()
        assert "command not implemented yet" not in captured.err

    def test_bench_missing_subcommand_returns_2(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

        code = main(["bench"])

        assert code == 2
        captured = capsys.readouterr()
        assert "bench" in captured.err
        assert "subcommand" in captured.err.lower()
        assert "command not implemented yet" not in captured.err

    def test_main_does_not_raise_system_exit_for_missing_subcommand(self, monkeypatch, tmp_path):
        """main() should return int 2, not raise SystemExit for missing subcommands."""
        monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

        # If main() raises SystemExit, this will propagate and pytest will fail
        code = main(["toolchain"])
        assert isinstance(code, int)
        assert code == 2

        code = main(["kernels"])
        assert isinstance(code, int)
        assert code == 2

        code = main(["bench"])
        assert isinstance(code, int)
        assert code == 2
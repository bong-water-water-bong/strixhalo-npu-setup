from npu_control_plane.benchmark import BenchmarkRecorder
from npu_control_plane.cli import main
from npu_control_plane.metadata import MetadataStore


def test_cli_help_returns_zero(capsys):
    code = main(["--help"])

    captured = capsys.readouterr()
    assert code == 0
    assert "npu-ctrl" in captured.out
    assert "discover" in captured.out


def test_benchmark_recorder_records_command(monkeypatch, tmp_path):
    times = iter([10.0, 10.1, 20.0, 20.2, 30.0, 30.4])

    def fake_perf_counter():
        return next(times)

    calls = []

    def fake_run_command(command, timeout=300):
        calls.append(list(command))
        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return Result()

    monkeypatch.setattr("npu_control_plane.benchmark.perf_counter", fake_perf_counter)
    monkeypatch.setattr("npu_control_plane.benchmark.run_command", fake_run_command)
    recorder = BenchmarkRecorder(MetadataStore(tmp_path / "store"))

    record = recorder.record_command(label="echo-ok", command=["echo", "ok"], warmup=1, iters=2)

    assert calls == [["echo", "ok"], ["echo", "ok"], ["echo", "ok"]]
    assert record["label"] == "echo-ok"
    assert record["warmup"] == 1
    assert record["iters"] == 2
    assert record["durations_ms"] == [199.9999999999993, 399.9999999999986]
    assert record["returncode"] == 0
    # list_runs returns lightweight summary entries
    summary = recorder.list_runs()[0]
    assert summary["run_id"] == record["run_id"]
    assert summary["label"] == "echo-ok"
    assert summary["timestamp"] == record["timestamp"]
    assert summary["median_ms"] == record["median_ms"]
    assert summary["returncode"] == 0
    assert summary["warmup"] == 1
    assert summary["iters"] == 2
    assert summary["command"] == ["echo", "ok"]
    # Full data stays in per-run file
    assert "stdout_tail" not in summary
    assert "stderr_tail" not in summary
    assert "durations_ms" not in summary


def test_cli_bench_list_uses_store_env(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

    code = main(["bench", "list"])

    captured = capsys.readouterr()
    assert code == 0
    assert '"runs": []' in captured.out


def test_cli_bench_run_records_and_returns_zero(monkeypatch, tmp_path, capsys):
    times = iter([10.0, 10.1])

    def fake_perf_counter():
        return next(times)

    class FakeResult:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    monkeypatch.setattr("npu_control_plane.benchmark.perf_counter", fake_perf_counter)
    monkeypatch.setattr("npu_control_plane.benchmark.run_command", lambda *a, **kw: FakeResult())
    monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

    code = main(["bench", "run", "--label", "x", "--warmup", "0", "--iters", "1", "--", "echo", "ok"])

    captured = capsys.readouterr()
    assert code == 0
    output = json.loads(captured.out)
    assert output["label"] == "x"
    assert output["returncode"] == 0
    assert "run_id" in output

    # Verify store contents
    store = MetadataStore(tmp_path / "store")
    summary = store.read_json("benchmarks", "summary.json")
    assert len(summary["runs"]) == 1
    run_summary = summary["runs"][0]
    assert run_summary["label"] == "x"
    assert "command" in run_summary
    assert "stdout_tail" not in run_summary


import json

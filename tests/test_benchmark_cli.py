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
    assert recorder.list_runs() == [record]


def test_cli_bench_list_uses_store_env(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NPU_CTRL_STORE", str(tmp_path / "store"))

    code = main(["bench", "list"])

    captured = capsys.readouterr()
    assert code == 0
    assert '"runs": []' in captured.out
from npu_control_plane.discovery import discover_devices
from npu_control_plane.metadata import MetadataStore
from npu_control_plane.toolchains import probe_toolchains


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.args = []


def test_discover_devices_parses_ryzenai_from_xrt_smi(monkeypatch, tmp_path):
    def fake_run(args, timeout=5):
        if args[:2] == ["xrt-smi", "examine"]:
            return FakeResult(stdout="[0000:c6:00.1]  |RyzenAI-npu5  |\n")
        if args[:1] == ["lsmod"]:
            return FakeResult(stdout="amdxdna 69632 0\n")
        return FakeResult(returncode=1, stderr="unexpected command")

    monkeypatch.setattr("npu_control_plane.discovery.run_command", fake_run)
    store = MetadataStore(tmp_path / "store")

    report = discover_devices(store)

    assert report["devices"] == [
        {
            "id": 0,
            "bdf": "0000:c6:00.1",
            "name": "RyzenAI-npu5",
            "driver": "amdxdna",
            "runtime": "xrt",
            "tile_count": 32,
            "peak_tflops_claimed": 31.2,
        }
    ]
    assert store.read_json("devices.json")["devices"][0]["name"] == "RyzenAI-npu5"


def test_discover_devices_reports_missing_xrt_without_failure(monkeypatch, tmp_path):
    def fake_run(args, timeout=5):
        if args[:2] == ["xrt-smi", "examine"]:
            return FakeResult(returncode=127, stderr="xrt-smi: not found")
        if args[:1] == ["lsmod"]:
            return FakeResult(stdout="")
        return FakeResult(returncode=1)

    monkeypatch.setattr("npu_control_plane.discovery.run_command", fake_run)
    store = MetadataStore(tmp_path / "store")

    report = discover_devices(store)

    assert report["devices"] == []
    assert report["driver_loaded"] is False
    assert "xrt-smi" in report["warnings"][0]


def test_probe_toolchains_detects_peano_iron_and_chess(monkeypatch, tmp_path):
    monkeypatch.setenv("PEANO_INSTALL_DIR", "/opt/peano")
    monkeypatch.setenv("XILINXD_LICENSE_FILE", "/licenses/Xilinx.lic")

    def fake_which(name):
        return {
            "clang++": "/opt/peano/bin/clang++",
            "xchesscc": "/opt/chess/bin/xchesscc",
            "python3": "/usr/bin/python3",
        }.get(name)

    def fake_run(args, timeout=5):
        if args[0] == "/usr/bin/python3":
            return FakeResult(stdout="iron-ok\n")
        if args[0] == "/opt/peano/bin/clang++":
            return FakeResult(stdout="clang version 21.0.0\n")
        if args[0] == "/opt/chess/bin/xchesscc":
            return FakeResult(stdout="Chess Compiler\n")
        return FakeResult(returncode=1, stderr="unexpected")

    monkeypatch.setattr("npu_control_plane.toolchains.which", fake_which)
    monkeypatch.setattr("npu_control_plane.toolchains.run_command", fake_run)
    store = MetadataStore(tmp_path / "store")

    report = probe_toolchains(store)

    by_name = {item["name"]: item for item in report["toolchains"]}
    assert by_name["peano"]["available"] is True
    assert by_name["iron"]["available"] is True
    assert by_name["chess"]["available"] is True
    assert by_name["chess"]["public_clean_room_use"] == "benchmark reference only; do not redistribute artifacts"
    assert store.read_json("toolchains.json") == report


def test_probe_toolchains_reports_missing_optional_tools(monkeypatch, tmp_path):
    monkeypatch.delenv("PEANO_INSTALL_DIR", raising=False)
    monkeypatch.delenv("XILINXD_LICENSE_FILE", raising=False)
    monkeypatch.setattr("npu_control_plane.toolchains.which", lambda name: None)
    store = MetadataStore(tmp_path / "store")

    report = probe_toolchains(store)

    by_name = {item["name"]: item for item in report["toolchains"]}
    assert by_name["peano"]["available"] is False
    assert "clang++" in by_name["peano"]["reason"]
    assert by_name["iron"]["available"] is False
    assert by_name["chess"]["available"] is False
    assert by_name["chess"]["license_required"] is True
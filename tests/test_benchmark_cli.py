from npu_control_plane.cli import main


def test_cli_help_returns_zero(capsys):
    code = main(["--help"])

    captured = capsys.readouterr()
    assert code == 0
    assert "npu-ctrl" in captured.out
    assert "discover" in captured.out

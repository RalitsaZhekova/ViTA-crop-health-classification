from pathlib import Path

from prithvi_crop.train import _configured_last_checkpoint


def test_auto_resume_uses_only_configured_checkpoint_directory(
    tmp_path: Path,
) -> None:
    configured = tmp_path / "configured"
    unrelated = tmp_path / "logs" / "version_99"
    configured.mkdir()
    unrelated.mkdir(parents=True)
    (unrelated / "last.ckpt").write_bytes(b"unrelated")

    assert _configured_last_checkpoint(configured) is None

    expected = configured / "last.ckpt"
    expected.write_bytes(b"configured")
    assert _configured_last_checkpoint(configured) == expected

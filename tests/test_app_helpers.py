"""File-cache helpers from the Streamlit entrypoint. No server and no model."""

from pathlib import Path

import pytest

import app


def test_format_size_uses_binary_units():
    assert app.format_size(0) == "0 B"
    assert app.format_size(1023) == "1023 B"
    assert app.format_size(1024) == "1.0 KB"
    assert app.format_size(1536) == "1.5 KB"
    assert app.format_size(1024 * 1024) == "1.0 MB"
    assert app.format_size(5 * 1024 * 1024 * 1024) == "5.0 GB"


def test_save_upload_strips_path_and_cache_clear_counts_files(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(app, "UPLOAD_DIR", upload_dir)

    class Upload:
        def __init__(self, name, payload):
            self.name = name
            self._payload = payload

        def getbuffer(self):
            return self._payload

    saved = Path(app.save_upload(Upload("../secret/notes.md", b"# notes")))
    assert saved.parent == upload_dir
    assert saved.name == "notes.md"
    assert saved.read_bytes() == b"# notes"

    nested = upload_dir / "nested"
    nested.mkdir()
    (nested / "keep.txt").write_text("directory entries are left alone", encoding="utf-8")

    assert app.clear_upload_cache() == 1
    assert saved.exists() is False
    assert (nested / "keep.txt").exists()
    assert app.clear_upload_cache() == 0


def test_save_upload_rejects_an_empty_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path / "uploads")

    class Upload:
        name = ""

        def getbuffer(self):
            return b"x"

    with pytest.raises(ValueError, match="文件名无效"):
        app.save_upload(Upload())

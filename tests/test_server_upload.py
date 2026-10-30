import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile

from dptb_pilot.server import app as server_app


class TrackingBytesIO(io.BytesIO):
    def __init__(self, content: bytes):
        super().__init__(content)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


def test_upload_accepts_11_mb_model_in_chunks(tmp_path, monkeypatch):
    payload = b"p" * (11 * 1024 * 1024)
    source = TrackingBytesIO(payload)
    upload = UploadFile(source, filename="model.pth")
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))
    monkeypatch.setattr(server_app, "MAX_UPLOAD_SIZE", 512 * 1024 * 1024)

    response = asyncio.run(server_app.upload_file("session", [upload]))

    destination = tmp_path / "session" / "files" / "model.pth"
    assert destination.read_bytes() == payload
    assert response["uploaded_files"][0]["size"] == len(payload)
    assert max(source.read_sizes) == server_app.UPLOAD_CHUNK_SIZE


def test_oversized_upload_returns_413_without_replacing_existing_file(tmp_path, monkeypatch):
    files_dir = tmp_path / "session" / "files"
    files_dir.mkdir(parents=True)
    destination = files_dir / "model.pth"
    destination.write_bytes(b"existing model")
    upload = UploadFile(io.BytesIO(b"x" * (2 * 1024 * 1024 + 1)), filename="model.pth")
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))
    monkeypatch.setattr(server_app, "MAX_UPLOAD_SIZE", 2 * 1024 * 1024)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server_app.upload_file("session", [upload]))

    assert exc_info.value.status_code == 413
    assert destination.read_bytes() == b"existing model"
    assert not list(files_dir.glob("*.part"))
    assert not list(files_dir.glob(".*.part"))


def test_list_files_recurses_into_workspace_subdirectories(tmp_path, monkeypatch):
    nested = tmp_path / "session" / "files" / "outputs" / "bands"
    nested.mkdir(parents=True)
    (nested / "eigenvalues.npy").write_bytes(b"numpy")
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))

    response = asyncio.run(server_app.list_files("session"))

    assert response["files"][0]["name"] == "outputs/bands/eigenvalues.npy"

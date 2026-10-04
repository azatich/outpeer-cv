from pathlib import Path
import io
import hashlib
import json
import zipfile

import pytest

from PIL import Image

from src.data.download import extract_archive, fetch_archive, safe_path, sequence_group, valid_image


def test_source_sequence_frames_stay_together():
    assert sequence_group("BATCH_d07_img_6400.jpg") == sequence_group("batch_d07_img_580.jpg")
    assert sequence_group("batch_01_frame_31.jpg") == "batch_01"
    assert sequence_group("GOPR0024.JPG") == sequence_group("GOPR0035.JPG")


def test_cached_jpeg_must_decode_not_only_pass_header_verification(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (128, 128), "red").save(path)
    record = {"width": 128, "height": 128}
    assert valid_image(path, record)
    path.write_bytes(path.read_bytes()[:-8])
    assert not valid_image(path, record)


def test_source_cannot_escape_download_directory(tmp_path: Path):
    for filename in ("../outside.jpg", "../../bad.jpg", str(tmp_path.parent / "bad.jpg")):
        with pytest.raises(ValueError):
            safe_path(tmp_path, filename)


def test_archive_ranges_resume_and_reconstruct_exact_bytes(tmp_path, monkeypatch):
    payload = b"abcdefghijklmnopqrstuvwxyz0123456789"
    requests = []

    class Response(io.BytesIO):
        status = 206

        def __init__(self, start, end):
            super().__init__(payload[start:end + 1])
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}

    def fake_open(request, timeout):
        header = request.get_header("Range")
        requests.append(header)
        start, end = map(int, header.removeprefix("bytes=").split("-"))
        return Response(start, end)

    monkeypatch.setattr("src.data.download.urlopen", fake_open)
    target = tmp_path / "data.zip"
    target.with_suffix(".zip.part").write_bytes(payload[:5])
    fetch_archive("https://example.test/data.zip", target, workers=4)
    assert target.read_bytes() == payload
    assert "bytes=5-8" in requests  # The first five bytes were reused.


def test_archive_refuses_unverified_range_response(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}

    monkeypatch.setattr("src.data.download.urlopen", lambda *a, **k: Response(b"unexpected full body"))
    target = tmp_path / "data.zip"
    with pytest.raises(RuntimeError, match="verified range"):
        fetch_archive("https://example.test/data.zip", target)
    assert not target.exists()


def test_archive_replaces_valid_but_different_cached_image(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cached = image_dir / "frame.png"
    Image.new("RGB", (16, 16), "blue").save(cached)
    original = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(original, format="PNG")
    archive = tmp_path / "UAVVasteDataset.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("source/frame.png", original.getvalue())
    (tmp_path / "annotations.json").write_text(json.dumps({
        "images": [{"file_name": "frame.png", "width": 16, "height": 16}]}))
    monkeypatch.setattr("src.data.download.ARCHIVE_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    extract_archive(tmp_path)
    assert cached.read_bytes() == original.getvalue()

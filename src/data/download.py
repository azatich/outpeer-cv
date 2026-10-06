"""Download the original UAVVaste COCO annotations and images, with provenance.

Run from the repository root: python -m src.data.download
Source filenames are also used to suggest *sequence* groups, not verified flights.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import threading
import zipfile
import zlib
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

SOURCE = "https://raw.githubusercontent.com/PUTvision/UAVVaste/main/annotations/annotations.json"
ARCHIVE = "https://zenodo.org/records/8214061/files/UAVVasteDataset.zip?download=1"
ARCHIVE_MD5 = "1575c32c04bdf944047563e4a1786c2a"


def safe_path(root: Path, filename: str) -> Path:
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Unsafe source filename: {filename!r}")
    return path


def fetch(url: str, destination: Path, attempts: int = 3) -> None:
    if urlparse(url).scheme != "https":
        raise ValueError("Downloads require an HTTPS URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "outpeer-cv-coursework/1.0"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                received = 0
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    received += len(chunk)
                    if received % (100 * 1024 * 1024) == 0:
                        print(f"Downloading {destination.name}: {received // (1024 * 1024)} MiB", flush=True)
            temporary.replace(destination)
            return
        except HTTPError as exc:
            if exc.code in {400, 401, 403, 404, 429} or attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(1 + attempt)


def sequence_group(filename: str) -> str:
    """Conservative filename grouping. Never claim these are known flight IDs."""
    stem = Path(filename).stem.lower()
    match = re.match(r"^(batch_[a-z]?\d+)_(?:img|frame)_?\d+$", stem)
    if match:
        return match.group(1)
    # Without frame metadata, keep each camera/name family together.
    if stem.startswith("gopr"):
        return "unverified_gopro"
    if stem.startswith("dji"):
        return "unverified_dji"
    if stem.startswith("photo_"):
        return "unverified_photo"
    if stem.startswith("camera"):
        return "unverified_camera"
    return "unverified_other"


def valid_image(path: Path, record: dict) -> bool:
    try:
        with Image.open(path) as image:
            if image.size != (record["width"], record["height"]):
                return False
            image.verify()
        # verify() checks JPEG structure but may not decode truncated pixel data.
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError):
        return False


def download(output: Path, limit: int | None = None, workers: int = 6,
             annotations_only: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    annotations = output / "annotations.json"
    if not annotations.exists():
        fetch(SOURCE, annotations)
    raw = annotations.read_bytes()
    data = json.loads(raw)
    records = sorted(data["images"], key=lambda item: item["id"])
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        groups: dict[str, list[dict]] = {}
        for record in records:
            groups.setdefault(sequence_group(record["file_name"]), []).append(record)
        selected = []
        while len(selected) < min(limit, len(records)):
            for group in groups.values():
                if group and len(selected) < limit:
                    selected.append(group.pop(0))
        records = sorted(selected, key=lambda item: item["id"])
    selected_ids = {record["id"] for record in records}
    subset = dict(data)
    subset["images"] = records
    subset["annotations"] = [a for a in data["annotations"] if a["image_id"] in selected_ids]
    subset_path = output / ("annotations_subset.json" if limit else "annotations.json")
    if limit:
        subset_path.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    with (output / "groups_suggested.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "group_id", "file_name", "basis"])
        writer.writeheader()
        for record in records:
            writer.writerow({"image_id": record["id"], "group_id": sequence_group(record["file_name"]),
                             "file_name": record["file_name"], "basis": "filename_heuristic_unverified"})
    failures = []
    rate_limited = threading.Event()

    def get_one(record: dict) -> None:
        destination = safe_path(output / "images", record["file_name"])
        if valid_image(destination, record):
            return
        if rate_limited.is_set():
            raise RuntimeError("Skipped after image server rate limit; use --source archive")
        url = record.get("flickr_url") or record.get("coco_url")
        if not url:
            raise ValueError("No image URL")
        try:
            fetch(url, destination)
        except HTTPError as exc:
            if exc.code == 429:
                rate_limited.set()
            raise
        if not valid_image(destination, record):
            raise ValueError("Downloaded file is invalid or dimensions differ from annotations")

    if not annotations_only:
        # Interleave sequences so an interrupted download still spans several scenes.
        queues: dict[str, list[dict]] = {}
        for record in records:
            queues.setdefault(sequence_group(record["file_name"]), []).append(record)
        ordered = []
        while any(queues.values()):
            for queue in queues.values():
                if queue:
                    ordered.append(queue.pop(0))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(get_one, record): record for record in ordered}
            for index, future in enumerate(as_completed(pending), 1):
                record = pending[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append({"image_id": record["id"], "file_name": record["file_name"], "error": str(exc)})
                    if len(failures) <= 5:
                        print(f"Download failed: {record['file_name']}: {exc}", flush=True)
                if index % 25 == 0 or index == len(records):
                    print(f"Images checked: {index}/{len(records)}; failures: {len(failures)}", flush=True)
    manifest = {
        "source": SOURCE, "paper": "https://doi.org/10.3390/rs13050965",
        "archive": "https://doi.org/10.5281/zenodo.8214061",
        "source_annotations_sha256": hashlib.sha256(raw).hexdigest(),
        "annotations_file": subset_path.name, "selected_images": len(records),
        "total_source_images": len(data["images"]), "download_failures": failures,
        "annotations_only": annotations_only,
        "grouping": "Filename sequence heuristic; not verified flight/location identifiers. Review groups before final evaluation.",
        "domain": "Aerial/general outdoor litter. Coastal or marine performance is not established by this dataset.",
    }
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} images failed. Re-run to retry; see download_manifest.json.")
    return manifest


def extract_archive(output: Path) -> None:
    """Use the authors' archive to avoid hundreds of requests to the image host."""
    archive = output / "UAVVasteDataset.zip"
    if not archive.exists():
        fetch_archive(ARCHIVE, archive)
    digest = hashlib.md5()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError("Archive checksum differs from the authors' Zenodo record. Keep it for inspection; do not extract.")
    annotations = output / "annotations.json"
    if not annotations.exists():
        fetch(SOURCE, annotations)
    records = json.loads(annotations.read_text(encoding="utf-8"))["images"]
    with zipfile.ZipFile(archive) as zipped:
        by_name = {}
        for info in zipped.infolist():
            if not info.is_dir() and Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                name = Path(info.filename).name
                if name in by_name:
                    raise ValueError(f"Duplicate basename in source archive: {name}")
                by_name[name] = info
        for index, record in enumerate(records, 1):
            destination = safe_path(output / "images", record["file_name"])
            info = by_name.get(record["file_name"])
            if info is None:
                raise ValueError(f"Source archive lacks {record['file_name']}")
            if destination.is_file() and destination.stat().st_size == info.file_size:
                checksum = 0
                with destination.open("rb") as cached:
                    for block in iter(lambda: cached.read(1024 * 1024), b""):
                        checksum = zlib.crc32(block, checksum)
                if checksum == info.CRC and valid_image(destination, record):
                    continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with zipped.open(info) as source, temporary.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            if not valid_image(temporary, record):
                raise ValueError(f"Invalid image in archive: {record['file_name']}")
            temporary.replace(destination)
            if index % 100 == 0:
                print(f"Archive images verified: {index}/{len(records)}", flush=True)


def fetch_archive(url: str, destination: Path, workers: int = 4) -> None:
    """Resume a large archive using verified HTTP byte ranges (four connections)."""
    probe = Request(url, headers={"Range": "bytes=0-0", "User-Agent": "outpeer-cv-coursework/1.0"})
    with urlopen(probe, timeout=60) as response:
        header = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-0/(\d+)", header)
        if response.status != 206 or not match:
            raise RuntimeError("Archive server does not support verified range downloads; retry later.")
        total = int(match.group(1))
        response.read(1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = (total + workers - 1) // workers
    partial_dir = destination.with_suffix(".chunks")
    partial_dir.mkdir(exist_ok=True)
    serial_partial = destination.with_suffix(destination.suffix + ".part")
    # Reuse any prefix obtained by the earlier sequential download.
    if serial_partial.exists() and not any(partial_dir.iterdir()):
        with serial_partial.open("rb") as source:
            remaining = min(serial_partial.stat().st_size, total)
            for index in range(workers):
                amount = min(chunk_size, remaining)
                if not amount:
                    break
                with (partial_dir / str(index)).open("wb") as target:
                    while amount:
                        block = source.read(min(amount, 1024 * 1024))
                        if not block:
                            raise RuntimeError("Partial archive shortened unexpectedly")
                        target.write(block)
                        amount -= len(block)
                        remaining -= len(block)

    def part(index: int) -> None:
        start, end = index * chunk_size, min((index + 1) * chunk_size, total) - 1
        expected = end - start + 1
        path = partial_dir / str(index)
        size = path.stat().st_size if path.exists() else 0
        if size > expected:
            raise RuntimeError(f"Oversized partial archive segment {index}")
        if size == expected:
            return
        headers = {"Range": f"bytes={start + size}-{end}", "User-Agent": "outpeer-cv-coursework/1.0"}
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start + size}-{end}/{total}":
                raise RuntimeError("Server returned an unexpected byte range; no bytes appended")
            with path.open("ab") as target:
                while block := response.read(min(1024 * 1024, expected - size + 1)):
                    if size + len(block) > expected:
                        raise RuntimeError("Archive range exceeds declared size")
                    target.write(block)
                    size += len(block)
                    if size // (100 * 1024 * 1024) != (size - len(block)) // (100 * 1024 * 1024):
                        print(f"Archive segment {index + 1}/{workers}: {size // (1024 * 1024)} MiB", flush=True)
        if size != expected:
            raise RuntimeError(f"Incomplete archive segment {index}; rerun to resume")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(part, range(workers)))
    with serial_partial.open("wb") as target:
        for index in range(workers):
            with (partial_dir / str(index)).open("rb") as source:
                shutil.copyfileobj(source, target, 1024 * 1024)
    serial_partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/uavvaste"))
    parser.add_argument("--limit", type=int, help="Representative subset for a quick pipeline check")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--source", choices=("archive", "images"), default="archive",
                        help="Authors' Zenodo archive (default), or individual original image URLs")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if args.source == "archive" and not args.annotations_only:
        extract_archive(args.output)
    manifest = download(args.output, args.limit, args.workers, args.annotations_only)
    print(json.dumps({key: manifest[key] for key in ("selected_images", "annotations_file", "grouping")}, indent=2))


if __name__ == "__main__":
    main()

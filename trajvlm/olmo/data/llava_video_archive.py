"""Resume-safe extraction of LLaVA-Video archives."""

import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


ACADEMIC_SUBSETS = (
    "0_30_s_academic_v0_1",
    "30_60_s_academic_v0_1",
    "1_2_m_academic_v0_1",
    "2_3_m_academic_v0_1",
)


def extract_archive(archive: Path) -> dict:
    """Extract members atomically; return a durable per-archive progress record."""
    archive = Path(archive)
    destination = archive.parent
    marker_dir = destination / ".stage2_extract"
    marker_dir.mkdir(exist_ok=True)
    marker = marker_dir / (archive.name + ".json")
    stat = archive.stat()
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get("archive_size") == stat.st_size and old.get("archive_mtime_ns") == stat.st_mtime_ns:
            return old

    extracted = skipped = 0
    total_bytes = 0
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            target = destination.joinpath(*name.parts)
            if target.is_file() and target.stat().st_size == member.size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f".part.{os.getpid()}")
            try:
                with source.extractfile(member) as src, temporary.open("wb") as dst:
                    while chunk := src.read(8 * 1024 * 1024):
                        dst.write(chunk)
                if temporary.stat().st_size != member.size:
                    raise IOError(f"Truncated archive member: {member.name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted += 1
            total_bytes += member.size
            if extracted % 500 == 0:
                print(f"{archive.name}: extracted={extracted} skipped={skipped}", flush=True)

    result = {
        "archive": str(archive),
        "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
        "extracted": extracted,
        "skipped": skipped,
        "extracted_bytes": total_bytes,
    }
    temporary_marker = marker.with_name(marker.name + f".part.{os.getpid()}")
    temporary_marker.write_text(json.dumps(result, indent=2) + "\n")
    temporary_marker.replace(marker)
    return result


def extract_zip_archive(archive: Path, destination: Path) -> dict:
    """Extract a ZIP into destination without leaving partial video files."""
    archive = Path(archive)
    destination = Path(destination)
    marker_dir = destination / ".stage2_extract"
    marker_dir.mkdir(exist_ok=True)
    marker = marker_dir / (archive.name + ".json")
    stat = archive.stat()
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get("archive_size") == stat.st_size and old.get("archive_mtime_ns") == stat.st_mtime_ns:
            return old

    extracted = skipped = 0
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                continue
            target = destination.joinpath(*name.parts)
            if target.is_file() and target.stat().st_size == member.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f".part.{os.getpid()}")
            try:
                with source.open(member) as src, temporary.open("wb") as dst:
                    while chunk := src.read(8 * 1024 * 1024):
                        dst.write(chunk)
                if temporary.stat().st_size != member.file_size:
                    raise IOError(f"Truncated archive member: {member.filename}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted += 1
            if extracted % 500 == 0:
                print(f"{archive.name}: extracted={extracted} skipped={skipped}", flush=True)

    result = {
        "archive": str(archive),
        "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
        "extracted": extracted,
        "skipped": skipped,
    }
    temporary_marker = marker.with_name(marker.name + f".part.{os.getpid()}")
    temporary_marker.write_text(json.dumps(result, indent=2) + "\n")
    temporary_marker.replace(marker)
    return result


def extract_academic_archives(root: Path, subsets=ACADEMIC_SUBSETS) -> list[dict]:
    results = []
    for subset in subsets:
        archives = sorted((Path(root) / subset).glob("*videos*.tar.gz"))
        if not archives:
            raise FileNotFoundError(f"No video archives in {Path(root) / subset}")
        for archive in archives:
            print(f"Starting {archive}", flush=True)
            result = extract_archive(archive)
            print(f"Finished {archive.name}: {result}", flush=True)
            results.append(result)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="LLaVA-Video-178K dataset directory")
    args = parser.parse_args()
    extract_academic_archives(args.root)

import io
import tarfile
import zipfile

from olmo.data.llava_video_archive import extract_archive, extract_zip_archive


def test_extract_archive_resumes_and_replaces_truncated_member(tmp_path):
    archive = tmp_path / "videos.tar.gz"
    payload = b"video contents" * 100
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("academic_source/sample.mp4")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    first = extract_archive(archive)
    target = tmp_path / "academic_source/sample.mp4"
    assert first["extracted"] == 1
    assert target.read_bytes() == payload

    target.write_bytes(b"incomplete")
    (tmp_path / ".stage2_extract/videos.tar.gz.json").unlink()
    resumed = extract_archive(archive)
    assert resumed["extracted"] == 1
    assert target.read_bytes() == payload
    assert not list(target.parent.glob("*.part.*"))


def test_extract_zip_archive_resumes(tmp_path):
    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("train/train_humor/sample.mp4", b"video contents")

    first = extract_zip_archive(archive, tmp_path)
    target = tmp_path / "train/train_humor/sample.mp4"
    assert first["extracted"] == 1
    assert target.read_bytes() == b"video contents"

    target.write_bytes(b"bad")
    (tmp_path / ".stage2_extract/train.zip.json").unlink()
    resumed = extract_zip_archive(archive, tmp_path)
    assert resumed["extracted"] == 1
    assert target.read_bytes() == b"video contents"

import json

import pytest

from olmo.data import academic_video_datasets
from olmo.data.academic_video_datasets import FunQA, SSv2QA


def test_ssv2_qa_resolves_relative_video_paths(tmp_path, monkeypatch):
    qa_path = tmp_path / "ssv2_qa_train.jsonl"
    rows = [
        {"video": "20bn-something-something-v2/1.webm", "style": "video_short_answer",
         "question": "q", "answer": "a"},
        {"video": "/abs/2.webm", "style": "video_short_answer", "question": "q", "answer": "a"},
    ]
    qa_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(SSv2QA, "HOME", str(tmp_path))
    monkeypatch.setattr(SSv2QA, "QA_PATH", str(qa_path))

    videos = [ex["video"] for ex in SSv2QA("train").data]
    assert videos == [str(tmp_path / "20bn-something-something-v2/1.webm"), "/abs/2.webm"]


def test_funqa_download_skips_extracted_split_without_archive(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    for name in ["FunQA_train.json", "Funqa_mcqa_v1.json"]:
        (tmp_path / "raw" / name).write_text("[]")
    (tmp_path / "train" / "train_humor").mkdir(parents=True)
    monkeypatch.setattr(FunQA, "HOME", str(tmp_path))

    def no_network(**kwargs):
        pytest.fail(f"unexpected download: {kwargs['filename']}")

    monkeypatch.setattr(academic_video_datasets, "hf_hub_download", no_network)
    FunQA.download(splits=("train",))

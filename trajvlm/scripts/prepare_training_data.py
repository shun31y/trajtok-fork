"""Download Stage1/Stage2 data and persist per-dataset results on the data disk.

A successful download method does not guarantee all externally hosted videos
exist. Consult each dataset's instructions and validate media before training.
"""
import argparse
import ast
import inspect
import json
import logging
import os
from pathlib import Path
import time


def dataset_names(stage):
    if stage == "pixmo_cap":
        return ["pixmo_cap"]
    # Read the authoritative mixture without importing the GPU training stack.
    source = Path(__file__).resolve().parents[1] / "launch_scripts/sft.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "VIDEO_ACADEMIC_DATASETS"
            for t in node.targets
        ):
            return list(dict.fromkeys(x[0] if isinstance(x, tuple) else x
                                     for x in ast.literal_eval(node.value)))
    raise RuntimeError("VIDEO_ACADEMIC_DATASETS missing")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["pixmo_cap", "video_academic"])
    parser.add_argument("--n-procs", type=int, default=8)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    names = dataset_names(args.stage)
    if args.list:
        print("\n".join(names))
        return 0
    root = Path(os.environ["MOLMO_DATA_DIR"])
    report_dir = root / "setup_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"{args.stage}.json"
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(),
        logging.FileHandler(report_dir / f"{args.stage}.log"),
    ])
    from olmo.data.get_dataset import get_dataset_class_by_name
    results = []
    seen = set()

    def save_report():
        tmp = report.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(report)

    for name in names:
        entry = {"dataset": name, "started": time.time(), "status": "running"}
        results.append(entry)
        save_report()
        try:
            cls = get_dataset_class_by_name(name)
            entry["class"] = cls.__name__
            if cls in seen:
                entry["status"] = "shared_download"
                continue
            params = inspect.signature(cls.download).parameters
            kwargs = {}
            if "n_procs" in params:
                kwargs["n_procs"] = args.n_procs
            elif "num_procs" in params:
                kwargs["num_procs"] = args.n_procs
            cls.download(**kwargs)
            seen.add(cls)
            entry["status"] = "download_method_returned"
        except Exception as error:
            logging.exception("Download failed: %s", name)
            entry.update(status="failed", error=str(error))
        finally:
            entry["finished"] = time.time()
            save_report()
    return int(any(x["status"] == "failed" for x in results))


if __name__ == "__main__":
    raise SystemExit(main())

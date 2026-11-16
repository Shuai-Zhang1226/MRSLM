from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.camus import iter_records, load_case, read_patients
from src.utils import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "configs" / "experiment.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = project_path(cfg, cfg["data"]["root"])
    report = {"root": str(root), "splits": {}}
    for split in ("train", "val", "test"):
        patients = read_patients(root, split)
        records = list(iter_records(root, split))
        report["splits"][split] = {"patients": len(patients), "frames": len(records)}
    labels = set()
    first = next(iter_records(root, "test"))
    _, lv, spacing = load_case(first, cfg["data"]["lv_label"])
    labels.add(int(lv.max()))
    report["sample"] = {"id": first.sample_id, "shape": list(lv.shape), "spacing_mm": spacing, "binary_labels": sorted(labels)}
    output = PROJECT / "data" / "dataset_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()

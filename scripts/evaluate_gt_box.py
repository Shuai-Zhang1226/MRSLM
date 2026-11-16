from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.camus import image_to_sam_tensor, iter_records, load_case, mask_to_box, pad_box, scale_box
from src.evaluation import metric_row, save_overlay, summarize_metrics
from src.models import SamEcho
from src.utils import load_config, project_path, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce SAM-Echo inference using ground-truth LV boxes.")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "experiment.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test on the first N frames.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    data_root = project_path(cfg, cfg["data"]["root"])
    checkpoint = project_path(cfg, cfg["model"]["checkpoint"])
    output = project_path(cfg, cfg["output"]["gt_box"])
    output.mkdir(parents=True, exist_ok=True)
    model = SamEcho(
        checkpoint,
        cfg["model"]["type"],
        cfg["device"],
        cfg["model"]["amp"],
        cfg["model"]["input_size"],
    )
    records = list(iter_records(data_root, "test", tuple(cfg["data"]["views"]), tuple(cfg["data"]["phases"])))
    if args.limit:
        records = records[: args.limit]
    rows = []
    started = time.perf_counter()
    for index, record in enumerate(tqdm(records, desc="GT Box evaluation")):
        image, target, spacing = load_case(record, cfg["data"]["lv_label"])
        gt_box = mask_to_box(target)
        prompt_box = pad_box(gt_box, image.shape, cfg["model"]["gt_box_padding"])
        image_tensor = image_to_sam_tensor(image, cfg["model"]["input_size"])
        box_1024 = scale_box(prompt_box, image.shape, cfg["model"]["input_size"])
        pred, _ = model.predict(image_tensor, box_1024, image.shape)
        rows.append(metric_row(record, pred, target, spacing))
        if index < 8:
            save_overlay(image, target, pred, prompt_box, output / "qualitative" / f"{record.sample_id}.png")
    elapsed = time.perf_counter() - started
    per_case = pd.DataFrame(rows)
    summary = summarize_metrics(per_case)
    per_case.to_csv(output / "per_frame_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": model.checkpoint_metadata,
        "protocol": "CAMUS official test split; LV label 1; A2C/A4C patient mean; ED/ES separate; tight GT box",
        "frames": len(records),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
    }
    save_json(metadata, output / "run_metadata.json")
    print(summary.to_string(index=False))
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"peak_gpu_memory_gb={metadata['peak_gpu_memory_gb']:.2f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.camus import image_to_sam_tensor, iter_records, load_case, mask_to_box, pad_box, unscale_box
from src.evaluation import metric_row, save_overlay, summarize_metrics
from src.metrics import box_iou
from src.models import AutoBoxHead, SamEcho
from src.models.autobox import decode_boxes
from src.utils import load_config, project_path, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM-Echo with automatically predicted box prompts.")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "experiment.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    settings = cfg["autobox"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    sam = SamEcho(
        project_path(cfg, cfg["model"]["checkpoint"]),
        cfg["model"]["type"],
        str(device),
        cfg["model"]["amp"],
        cfg["model"]["input_size"],
    )
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else project_path(cfg, settings["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    box_model = AutoBoxHead(hidden_channels=settings["hidden_channels"], num_blocks=settings["num_blocks"]).to(device)
    box_model.load_state_dict(payload["model"])
    box_model.eval()
    data_root = project_path(cfg, cfg["data"]["root"])
    output = project_path(cfg, cfg["output"]["autobox"])
    output.mkdir(parents=True, exist_ok=True)
    records = list(iter_records(data_root, "test", tuple(cfg["data"]["views"]), tuple(cfg["data"]["phases"])))
    if args.limit:
        records = records[: args.limit]
    rows = []
    started = time.perf_counter()
    for index, record in enumerate(tqdm(records, desc="AutoBox evaluation")):
        image, target, spacing = load_case(record, cfg["data"]["lv_label"])
        image_tensor = image_to_sam_tensor(image, cfg["model"]["input_size"])
        embedding = sam.encode(image_tensor)
        with torch.inference_mode():
            center, ltrb = box_model(embedding)
            predicted_1024 = decode_boxes(
                center,
                ltrb,
                cfg["model"]["input_size"],
                cfg["model"]["autobox_padding"],
            )[0].cpu().numpy()
        predicted_box = unscale_box(predicted_1024, image.shape, cfg["model"]["input_size"])
        target_box = pad_box(mask_to_box(target), image.shape, cfg["model"]["autobox_padding"])
        pred = sam.decode(embedding, predicted_1024, image.shape)
        rows.append(metric_row(record, pred, target, spacing, box_iou=box_iou(predicted_box, target_box)))
        if index < 8:
            save_overlay(image, target, pred, predicted_box, output / "qualitative" / f"{record.sample_id}.png")
    elapsed = time.perf_counter() - started
    per_case = pd.DataFrame(rows)
    summary = summarize_metrics(per_case)
    summary["box_iou_mean"] = per_case.groupby("phase")["box_iou"].mean().reindex(summary["phase"]).to_numpy()
    per_case.to_csv(output / "per_frame_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    save_json(
        {
            "sam_checkpoint": str(project_path(cfg, cfg["model"]["checkpoint"])),
            "autobox_checkpoint": str(checkpoint),
            "autobox_epoch": payload.get("epoch"),
            "autobox_val_dice": payload.get("val_dice"),
            "autobox_val_box_iou": payload.get("val_box_iou"),
            "training_loss": payload.get("loss"),
            "frames": len(records),
            "elapsed_seconds": elapsed,
        },
        output / "run_metadata.json",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

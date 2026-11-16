from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import cv2
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.camus import image_to_sam_tensor, iter_records, load_case, mask_to_box, scale_box
from src.models import SamEcho
from src.utils import load_config, project_path, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen SAM-Echo embeddings for AutoBox training.")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "experiment.yaml"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    data_root = project_path(cfg, cfg["data"]["root"])
    cache_root = project_path(cfg, cfg["cache"]["root"])
    model = SamEcho(
        project_path(cfg, cfg["model"]["checkpoint"]),
        cfg["model"]["type"],
        cfg["device"],
        cfg["model"]["amp"],
        cfg["model"]["input_size"],
    )
    for split in args.splits:
        output = cache_root / split
        output.mkdir(parents=True, exist_ok=True)
        records = list(iter_records(data_root, split, tuple(cfg["data"]["views"]), tuple(cfg["data"]["phases"])))
        if args.limit:
            records = records[: args.limit]
        for record in tqdm(records, desc=f"Caching {split}"):
            destination = output / f"{record.sample_id}.pt"
            if destination.exists() and not args.overwrite:
                payload = torch.load(destination, map_location="cpu", weights_only=False)
                if "target_mask_256" in payload:
                    continue
                _, target, _ = load_case(record, cfg["data"]["lv_label"])
                target_256 = cv2.resize(target.astype("uint8"), (256, 256), interpolation=cv2.INTER_NEAREST)
                payload["target_mask_256"] = torch.from_numpy(target_256).unsqueeze(0)
                torch.save(payload, destination)
                continue
            image, target, spacing = load_case(record, cfg["data"]["lv_label"])
            box = mask_to_box(target)
            box_1024 = scale_box(box, image.shape, cfg["model"]["input_size"])
            image_tensor = image_to_sam_tensor(image, cfg["model"]["input_size"])
            embedding = model.encode(image_tensor).cpu().half()
            target_256 = cv2.resize(target.astype("uint8"), (256, 256), interpolation=cv2.INTER_NEAREST)
            torch.save(
                {
                    "embedding": embedding,
                    "box_1024": torch.from_numpy(box_1024),
                    "sample_id": record.sample_id,
                    "record": record.to_dict(),
                    "image_shape": image.shape,
                    "spacing": spacing,
                    "target_mask_256": torch.from_numpy(target_256).unsqueeze(0),
                },
                destination,
            )
    print(f"cache_root={cache_root}")


if __name__ == "__main__":
    main()

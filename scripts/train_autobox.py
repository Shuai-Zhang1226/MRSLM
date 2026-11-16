from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.cache_dataset import EmbeddingDataset
from src.losses import autobox_loss, segmentation_loss
from src.models import AutoBoxHead, SamEcho
from src.models.autobox import decode_boxes, decode_boxes_soft
from src.utils import load_config, project_path, seed_everything


def tensor_box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp_min(0).prod(dim=1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(dim=1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(dim=1)
    return inter / (area_a + area_b - inter).clamp_min(1e-6)


def binary_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = torch.sigmoid(logits) > 0.5
    target = target > 0.5
    intersection = (prediction & target).sum(dim=(1, 2, 3)).float()
    denominator = prediction.sum(dim=(1, 2, 3)).float() + target.sum(dim=(1, 2, 3)).float()
    return (2.0 * intersection + 1.0) / (denominator + 1.0)


@torch.no_grad()
def validate(
    model: AutoBoxHead,
    sam: SamEcho,
    loader: DataLoader,
    device: torch.device,
    padding: float,
) -> tuple[float, float]:
    model.eval()
    dice_values: list[float] = []
    iou_values: list[float] = []
    for batch in loader:
        embedding = batch["embedding"].to(device, non_blocking=True)
        target_boxes = batch["box_1024"].to(device, non_blocking=True)
        target_masks = batch["target_mask_256"].to(device, non_blocking=True)
        center, ltrb = model(embedding)
        prompt_boxes = decode_boxes(center, ltrb, padding=padding)
        raw_boxes = decode_boxes(center, ltrb, padding=0.0)
        logits = sam.decode_logits_with_grad(embedding, prompt_boxes)
        dice_values.extend(binary_dice(logits, target_masks).cpu().tolist())
        iou_values.extend(tensor_box_iou(raw_boxes, target_boxes).cpu().tolist())
    return float(sum(dice_values) / len(dice_values)), float(sum(iou_values) / len(iou_values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train segmentation-aware AutoBox with frozen SAM-Echo.")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "experiment.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Resume AutoBox weights from a saved checkpoint.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    cache_root = project_path(cfg, cfg["cache"]["root"])
    train_set = EmbeddingDataset(cache_root, "train")
    val_set = EmbeddingDataset(cache_root, "val")
    settings = cfg["autobox"]
    train_loader = DataLoader(
        train_set,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        pin_memory=True,
    )
    model = AutoBoxHead(hidden_channels=settings["hidden_channels"], num_blocks=settings["num_blocks"]).to(device)
    sam = SamEcho(
        project_path(cfg, cfg["model"]["checkpoint"]),
        cfg["model"]["type"],
        str(device),
        amp=False,
        input_size=cfg["model"]["input_size"],
    )
    epochs = args.epochs or settings["epochs"]
    history_path = PROJECT / "logs" / "autobox_segaware_history.csv"
    start_epoch = 1
    best_dice = -1.0
    resume_payload = None
    resume_lr = settings["learning_rate"]
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["model"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_dice = float(resume_payload.get("val_dice", -1.0))
        if history_path.exists():
            import pandas as pd

            previous_history = pd.read_csv(history_path)
            matching = previous_history.loc[previous_history["epoch"] == resume_payload["epoch"]]
            if not matching.empty:
                resume_lr = float(matching.iloc[-1]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=resume_lr, weight_decay=settings["weight_decay"])
    if resume_payload and "optimizer" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - start_epoch + 1))
    writer = SummaryWriter(PROJECT / "logs" / "autobox_segaware")
    checkpoint = project_path(cfg, settings["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "train_total_loss",
        "train_segmentation_loss",
        "train_dice_loss",
        "train_bce_loss",
        "train_box_loss",
        "val_dice",
        "val_box_iou",
        "learning_rate",
    ]
    history_mode = "a" if args.resume and history_path.exists() else "w"
    with history_path.open(history_mode, newline="", encoding="utf-8") as history_file:
        history = csv.DictWriter(history_file, fieldnames=fields)
        if history_mode == "w":
            history.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            totals = {"total": 0.0, "segmentation": 0.0, "dice": 0.0, "bce": 0.0, "box": 0.0}
            for batch in tqdm(train_loader, desc=f"Seg-aware epoch {epoch}/{epochs}"):
                embedding = batch["embedding"].to(device, non_blocking=True)
                target_boxes = batch["box_1024"].to(device, non_blocking=True)
                target_masks = batch["target_mask_256"].to(device, non_blocking=True)
                center, ltrb = model(embedding)
                prompt_boxes = decode_boxes_soft(
                    center,
                    ltrb,
                    cfg["model"]["input_size"],
                    cfg["model"]["autobox_padding"],
                    settings["softargmax_temperature"],
                )
                mask_logits = sam.decode_logits_with_grad(embedding, prompt_boxes)
                seg_loss, seg_parts = segmentation_loss(mask_logits, target_masks)
                box_loss, _ = autobox_loss(
                    center,
                    ltrb,
                    target_boxes,
                    settings["center_loss_weight"],
                    settings["ltrb_loss_weight"],
                    settings["giou_loss_weight"],
                )
                loss = seg_loss + settings["box_aux_weight"] * box_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                batch_size = embedding.shape[0]
                totals["total"] += loss.item() * batch_size
                totals["segmentation"] += seg_parts["segmentation"] * batch_size
                totals["dice"] += seg_parts["dice"] * batch_size
                totals["bce"] += seg_parts["bce"] * batch_size
                totals["box"] += box_loss.item() * batch_size
            averages = {key: value / len(train_set) for key, value in totals.items()}
            val_dice, val_iou = validate(model, sam, val_loader, device, cfg["model"]["autobox_padding"])
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_total_loss": averages["total"],
                "train_segmentation_loss": averages["segmentation"],
                "train_dice_loss": averages["dice"],
                "train_bce_loss": averages["bce"],
                "train_box_loss": averages["box"],
                "val_dice": val_dice,
                "val_box_iou": val_iou,
                "learning_rate": lr,
            }
            history.writerow(row)
            history_file.flush()
            for key, value in averages.items():
                writer.add_scalar(f"loss/{key}", value, epoch)
            writer.add_scalar("dice/validation", val_dice, epoch)
            writer.add_scalar("box_iou/validation", val_iou, epoch)
            print(
                f"epoch={epoch} total={averages['total']:.5f} seg={averages['segmentation']:.5f} "
                f"box={averages['box']:.5f} val_dice={val_dice:.4f} val_box_iou={val_iou:.4f}"
            )
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_dice": val_dice,
                        "val_box_iou": val_iou,
                        "loss": "Dice + BCE + 0.2 * Box",
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": cfg,
                    },
                    checkpoint,
                )
    writer.close()
    print(f"best_val_dice={best_dice:.4f}")
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()

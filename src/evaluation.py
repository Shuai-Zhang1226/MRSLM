from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .metrics import dice_score, hd95_score, iou_score


def metric_row(record, pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float], **extra) -> dict:
    return {
        "sample_id": record.sample_id,
        "patient": record.patient,
        "view": record.view,
        "phase": record.phase,
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "hd95_mm": hd95_score(pred, target, spacing),
        **extra,
    }


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    # Each patient/phase contributes one value after averaging its A2C and A4C frames.
    patient_phase = frame.groupby(["patient", "phase"], as_index=False)[["dice", "iou", "hd95_mm"]].mean()
    rows = []
    for phase, group in patient_phase.groupby("phase"):
        row = {"phase": phase, "n_patients": int(group["patient"].nunique())}
        for metric in ("dice", "iou", "hd95_mm"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)


def save_overlay(image: np.ndarray, target: np.ndarray, pred: np.ndarray, box: np.ndarray, path: str | Path) -> None:
    lo, hi = float(image.min()), float(image.max())
    gray = ((image - lo) / max(hi - lo, 1e-8) * 255).astype(np.uint8)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gt_contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, gt_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(canvas, pred_contours, -1, (0, 0, 255), 2)
    x0, y0, x1, y1 = np.round(box).astype(int)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 128, 0), 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)

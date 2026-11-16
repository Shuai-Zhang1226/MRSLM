from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    denom = pred.sum() + target.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(pred, target).sum() / denom)


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    union = np.logical_or(pred, target).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, target).sum() / union)


def hd95_score(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float]) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float(math.hypot(pred.shape[0] * spacing[0], pred.shape[1] * spacing[1]))
    pred_surface = np.logical_xor(pred, binary_erosion(pred))
    target_surface = np.logical_xor(target, binary_erosion(target))
    target_distance = distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([target_distance[pred_surface], pred_distance[target_surface]])
    return float(np.percentile(distances, 95))


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return inter / max(area_a + area_b - inter, 1e-8)

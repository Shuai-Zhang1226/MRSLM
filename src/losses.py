from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_heatmap(boxes: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Boxes are in grid coordinates; return heatmaps and integer center indices."""
    device = boxes.device
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    xs = torch.arange(size, device=device).float() + 0.5
    ys = torch.arange(size, device=device).float() + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
    height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
    sigma = (torch.minimum(width, height) / 6.0).clamp_min(1.0)
    distance = (xx[None] - centers[:, 0, None, None]).square() + (yy[None] - centers[:, 1, None, None]).square()
    heatmap = torch.exp(-distance / (2.0 * sigma[:, None, None].square())).unsqueeze(1)
    center_indices = torch.stack(
        [centers[:, 1].floor().long().clamp(0, size - 1), centers[:, 0].floor().long().clamp(0, size - 1)], dim=1
    )
    batch_indices = torch.arange(boxes.shape[0], device=device)
    heatmap[batch_indices, 0, center_indices[:, 0], center_indices[:, 1]] = 1.0
    return heatmap, center_indices


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    positive = target.eq(1.0)
    negative = target.lt(1.0)
    negative_weights = (1.0 - target).pow(4)
    positive_loss = torch.log(probability) * (1.0 - probability).pow(2) * positive
    negative_loss = torch.log(1.0 - probability) * probability.pow(2) * negative_weights * negative
    num_positive = positive.float().sum().clamp_min(1.0)
    return -(positive_loss.sum() + negative_loss.sum()) / num_positive


def generalized_box_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(pred[:, :2], target[:, :2])
    rb = torch.minimum(pred[:, 2:], target[:, 2:])
    inter = (rb - lt).clamp_min(0).prod(dim=1)
    pred_area = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(dim=1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(dim=1)
    union = (pred_area + target_area - inter).clamp_min(1e-6)
    iou = inter / union
    enc_lt = torch.minimum(pred[:, :2], target[:, :2])
    enc_rb = torch.maximum(pred[:, 2:], target[:, 2:])
    enc_area = (enc_rb - enc_lt).clamp_min(0).prod(dim=1).clamp_min(1e-6)
    giou = iou - (enc_area - union) / enc_area
    return (1.0 - giou).mean()


def autobox_loss(
    center_logits: torch.Tensor,
    ltrb: torch.Tensor,
    target_boxes_1024: torch.Tensor,
    center_weight: float = 1.0,
    ltrb_weight: float = 2.0,
    giou_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    size = center_logits.shape[-1]
    scale = 1024.0 / float(size)
    target_boxes = target_boxes_1024 / scale
    heatmap, indices = gaussian_heatmap(target_boxes, size)
    center_loss = focal_heatmap_loss(center_logits, heatmap)
    batch_indices = torch.arange(ltrb.shape[0], device=ltrb.device)
    pred_dist = ltrb.permute(0, 2, 3, 1)[batch_indices, indices[:, 0], indices[:, 1]]
    centers = torch.stack([indices[:, 1].float() + 0.5, indices[:, 0].float() + 0.5], dim=1)
    target_dist = torch.stack(
        [centers[:, 0] - target_boxes[:, 0], centers[:, 1] - target_boxes[:, 1], target_boxes[:, 2] - centers[:, 0], target_boxes[:, 3] - centers[:, 1]],
        dim=1,
    ).clamp_min(0)
    distance_loss = F.smooth_l1_loss(pred_dist / size, target_dist / size)
    pred_boxes = torch.stack(
        [centers[:, 0] - pred_dist[:, 0], centers[:, 1] - pred_dist[:, 1], centers[:, 0] + pred_dist[:, 2], centers[:, 1] + pred_dist[:, 3]],
        dim=1,
    )
    giou_loss = generalized_box_iou_loss(pred_boxes, target_boxes)
    total = center_weight * center_loss + ltrb_weight * distance_loss + giou_weight * giou_loss
    values = {"center": center_loss.item(), "ltrb": distance_loss.item(), "giou": giou_loss.item(), "total": total.item()}
    return total, values


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    target = target.float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    reduce_dims = (1, 2, 3)
    intersection = (probability * target).sum(dim=reduce_dims)
    denominator = probability.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice_loss = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = dice_loss + bce
    return total, {"dice": dice_loss.item(), "bce": bce.item(), "segmentation": total.item()}

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class AutoBoxHead(nn.Module):
    """Predict an LV center heatmap and L/T/R/B distances on the 64x64 SAM grid."""

    def __init__(self, in_channels: int = 256, hidden_channels: int = 128, num_blocks: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        ]
        layers.extend(ResidualBlock(hidden_channels) for _ in range(num_blocks))
        self.trunk = nn.Sequential(*layers)
        self.center_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )
        self.ltrb_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 4, 1),
        )
        nn.init.constant_(self.center_head[-1].bias, -2.19)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(embedding.float())
        center_logits = self.center_head(features)
        # Distances are expressed in 64x64 feature-grid cells. A bounded
        # parameterization starts from useful, non-degenerate boxes.
        ltrb = torch.sigmoid(self.ltrb_head(features)) * float(features.shape[-1])
        return center_logits, ltrb


def decode_boxes(center_logits: torch.Tensor, ltrb: torch.Tensor, input_size: int = 1024, padding: float = 0.05) -> torch.Tensor:
    batch, _, height, width = center_logits.shape
    indices = center_logits.flatten(1).argmax(dim=1)
    ys = torch.div(indices, width, rounding_mode="floor")
    xs = indices % width
    distances = ltrb.permute(0, 2, 3, 1)[torch.arange(batch, device=ltrb.device), ys, xs]
    centers_x = xs.float() + 0.5
    centers_y = ys.float() + 0.5
    boxes = torch.stack(
        [centers_x - distances[:, 0], centers_y - distances[:, 1], centers_x + distances[:, 2], centers_y + distances[:, 3]],
        dim=1,
    )
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] -= widths * padding
    boxes[:, 2] += widths * padding
    boxes[:, 1] -= heights * padding
    boxes[:, 3] += heights * padding
    boxes[:, [0, 2]].clamp_(0, width)
    boxes[:, [1, 3]].clamp_(0, height)
    scale = float(input_size) / float(width)
    return boxes * scale


def decode_boxes_soft(
    center_logits: torch.Tensor,
    ltrb: torch.Tensor,
    input_size: int = 1024,
    padding: float = 0.05,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable box decoding used only during segmentation-aware training."""
    batch, _, height, width = center_logits.shape
    probabilities = torch.softmax(center_logits.flatten(1) / temperature, dim=1).view(batch, 1, height, width)
    xs = torch.arange(width, device=center_logits.device, dtype=center_logits.dtype) + 0.5
    ys = torch.arange(height, device=center_logits.device, dtype=center_logits.dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    center_x = (probabilities[:, 0] * xx).sum(dim=(1, 2))
    center_y = (probabilities[:, 0] * yy).sum(dim=(1, 2))
    distances = (probabilities * ltrb).sum(dim=(2, 3))
    boxes = torch.stack(
        [center_x - distances[:, 0], center_y - distances[:, 1], center_x + distances[:, 2], center_y + distances[:, 3]],
        dim=1,
    )
    box_widths = boxes[:, 2] - boxes[:, 0]
    box_heights = boxes[:, 3] - boxes[:, 1]
    boxes = torch.stack(
        [
            boxes[:, 0] - box_widths * padding,
            boxes[:, 1] - box_heights * padding,
            boxes[:, 2] + box_widths * padding,
            boxes[:, 3] + box_heights * padding,
        ],
        dim=1,
    )
    # Avoid in-place clamping because it can invalidate autograd intermediates.
    x0 = boxes[:, 0].clamp(0, width)
    y0 = boxes[:, 1].clamp(0, height)
    x1 = boxes[:, 2].clamp(0, width)
    y1 = boxes[:, 3].clamp(0, height)
    scale = float(input_size) / float(width)
    return torch.stack([x0, y0, x1, y1], dim=1) * scale

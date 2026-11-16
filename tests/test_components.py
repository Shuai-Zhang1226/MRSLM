from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.camus import mask_to_box, pad_box, scale_box, unscale_box
from src.losses import autobox_loss
from src.metrics import dice_score, hd95_score, iou_score
from src.models import AutoBoxHead
from src.models.autobox import decode_boxes


def main() -> None:
    mask = np.zeros((100, 80), dtype=bool)
    mask[20:70, 10:50] = True
    box = mask_to_box(mask)
    assert np.allclose(box, [10, 20, 50, 70])
    padded = pad_box(box, mask.shape, 0.05)
    scaled = scale_box(padded, mask.shape)
    assert np.allclose(unscale_box(scaled, mask.shape), padded)
    assert dice_score(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0
    assert hd95_score(mask, mask, (0.3, 0.3)) == 0.0
    model = AutoBoxHead(hidden_channels=32, num_blocks=1)
    center, ltrb = model(torch.randn(2, 256, 64, 64))
    targets = torch.tensor([[100, 120, 500, 700], [250, 200, 800, 900]], dtype=torch.float32)
    loss, values = autobox_loss(center, ltrb, targets)
    loss.backward()
    boxes = decode_boxes(center.detach(), ltrb.detach())
    assert boxes.shape == (2, 4)
    assert torch.isfinite(loss)
    print("component_tests=passed", values)


if __name__ == "__main__":
    main()

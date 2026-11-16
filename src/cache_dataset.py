from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.files = sorted((Path(root) / split).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No cached embeddings found in {Path(root) / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        item = torch.load(self.files[index], map_location="cpu", weights_only=False)
        return {
            "embedding": item["embedding"].squeeze(0).float(),
            "box_1024": item["box_1024"].float(),
            "target_mask_256": item["target_mask_256"].float(),
            "sample_id": item["sample_id"],
        }

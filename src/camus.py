from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class CamusRecord:
    patient: str
    view: str
    phase: str
    image_path: str
    mask_path: str

    @property
    def sample_id(self) -> str:
        return f"{self.patient}_{self.view}_{self.phase}"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def split_file(root: Path, split: str) -> Path:
    mapping = {
        "train": "subgroup_training.txt",
        "val": "subgroup_validation.txt",
        "test": "subgroup_testing.txt",
    }
    if split not in mapping:
        raise ValueError(f"Unknown CAMUS split: {split}")
    return root / "database_split" / mapping[split]


def read_patients(root: str | Path, split: str) -> list[str]:
    root = Path(root)
    return [line.strip() for line in split_file(root, split).read_text().splitlines() if line.strip()]


def iter_records(
    root: str | Path,
    split: str,
    views: tuple[str, ...] = ("2CH", "4CH"),
    phases: tuple[str, ...] = ("ED", "ES"),
) -> Iterator[CamusRecord]:
    root = Path(root)
    for patient in read_patients(root, split):
        patient_dir = root / "database_nifti" / patient
        for view in views:
            for phase in phases:
                stem = f"{patient}_{view}_{phase}"
                image = patient_dir / f"{stem}.nii.gz"
                mask = patient_dir / f"{stem}_gt.nii.gz"
                if not image.exists() or not mask.exists():
                    raise FileNotFoundError(f"Missing CAMUS pair: {image}, {mask}")
                yield CamusRecord(patient, view, phase, str(image), str(mask))


def load_case(record: CamusRecord, lv_label: int = 1) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    image_nii = nib.load(record.image_path)
    mask_nii = nib.load(record.mask_path)
    image = np.asarray(image_nii.dataobj, dtype=np.float32).squeeze()
    labels = np.asarray(mask_nii.dataobj).squeeze()
    if image.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"Expected 2-D ED/ES frame for {record.sample_id}, got {image.shape}")
    mask = labels == lv_label
    zooms = mask_nii.header.get_zooms()[:2]
    spacing = (float(zooms[0]), float(zooms[1]))
    return image, mask, spacing


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Cannot create a box from an empty LV mask")
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def pad_box(box: np.ndarray, image_shape: tuple[int, int], fraction: float) -> np.ndarray:
    h, w = image_shape
    x0, y0, x1, y1 = box.astype(np.float32)
    dx = (x1 - x0) * fraction
    dy = (y1 - y0) * fraction
    return np.array(
        [max(0.0, x0 - dx), max(0.0, y0 - dy), min(float(w), x1 + dx), min(float(h), y1 + dy)],
        dtype=np.float32,
    )


def image_to_sam_tensor(image: np.ndarray, input_size: int = 1024) -> np.ndarray:
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(resized.min()), float(resized.max())
    resized = (resized - lo) / max(hi - lo, 1e-8)
    rgb = np.repeat(resized[..., None], 3, axis=2)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)


def scale_box(box: np.ndarray, image_shape: tuple[int, int], input_size: int = 1024) -> np.ndarray:
    h, w = image_shape
    return box / np.array([w, h, w, h], dtype=np.float32) * float(input_size)


def unscale_box(box: np.ndarray, image_shape: tuple[int, int], input_size: int = 1024) -> np.ndarray:
    h, w = image_shape
    return box / float(input_size) * np.array([w, h, w, h], dtype=np.float32)

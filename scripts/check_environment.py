from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import cv2
import nibabel
import numpy
import torch


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"opencv={cv2.__version__}")
    print(f"nibabel={nibabel.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"gpu={properties.name}")
        print(f"vram_gb={properties.total_memory / 2**30:.2f}")


if __name__ == "__main__":
    main()

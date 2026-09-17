"""Dataset loaders for the PNG layout used by the coursework."""

from __future__ import annotations

import os
from glob import glob

import cv2
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Load paired images and masks without changing their stored values."""

    def __init__(self, root: str = "") -> None:
        super().__init__()
        self.image_files = glob(os.path.join(root, "image", "*.png"))
        self.mask_files = [
            os.path.join(root, "mask", os.path.basename(path)[:-4] + "_mask.png")
            for path in self.image_files
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.image_files[index], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.mask_files[index], cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()

    def __len__(self) -> int:
        return len(self.image_files)


class TestImageDataset(Dataset):
    """Load unlabelled test images and their filename stems."""

    def __init__(self, root: str = "") -> None:
        super().__init__()
        self.image_files = glob(os.path.join(root, "image", "*.png"))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_files[index]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        filename = os.path.basename(image_path).rsplit(".", maxsplit=1)[0]
        return torch.from_numpy(image).float(), filename

    def __len__(self) -> int:
        return len(self.image_files)

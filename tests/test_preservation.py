from pathlib import Path

import cv2
import numpy as np
import torch

from unet_cardiac_segmentation import (
    SegmentationDataset,
    UNet,
    batch_foreground_dice,
)
from unet_cardiac_segmentation import (
    TestImageDataset as InferenceDataset,
)


def test_default_model_preserves_coursework_parameter_count() -> None:
    model = UNet()
    assert sum(parameter.numel() for parameter in model.parameters()) == 31_042_564


def test_foreground_dice_uses_classes_one_to_three() -> None:
    target = torch.tensor([[[0, 1], [2, 3]]])
    logits = torch.full((1, 4, 2, 2), -1.0)
    logits.scatter_(1, target.unsqueeze(1), 1.0)
    assert batch_foreground_dice(logits, target) == 1.0


def test_datasets_preserve_the_coursework_png_contract(tmp_path: Path) -> None:
    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "mask"
    image_dir.mkdir()
    mask_dir.mkdir()

    image = np.array([[0, 1024], [2048, 4095]], dtype=np.uint16)
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    cv2.imwrite(str(image_dir / "sample.png"), image)
    cv2.imwrite(str(mask_dir / "sample_mask.png"), mask)

    labelled = SegmentationDataset(str(tmp_path))
    loaded_image, loaded_mask = labelled[0]
    assert torch.equal(loaded_image, torch.from_numpy(image).float())
    assert torch.equal(loaded_mask, torch.from_numpy(mask).float())

    unlabelled = InferenceDataset(str(tmp_path))
    test_image, identifier = unlabelled[0]
    assert torch.equal(test_image, torch.from_numpy(image).float())
    assert identifier == "sample"

"""Foreground Dice metric used by the coursework."""

from __future__ import annotations

import numpy as np
from torch import Tensor


def categorical_dice(mask1: np.ndarray, mask2: np.ndarray, label_class: int = 1) -> float:
    """Calculate Dice for one integer-encoded class."""
    mask1_positive = (mask1 == label_class).astype(np.float32)
    mask2_positive = (mask2 == label_class).astype(np.float32)
    numerator = 2 * np.sum(mask1_positive * mask2_positive)
    denominator = np.sum(mask1_positive) + np.sum(mask2_positive)
    return float(numerator / denominator)


def mean_foreground_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Average Dice across the three foreground classes."""
    return float(
        np.mean(
            [
                categorical_dice(prediction, target, 1),
                categorical_dice(prediction, target, 2),
                categorical_dice(prediction, target, 3),
            ]
        )
    )


def batch_foreground_dice(prediction: Tensor, target: Tensor) -> float:
    """Average foreground Dice across a batch of class logits."""
    score = 0.0
    for index in range(prediction.shape[0]):
        predicted_mask = prediction[index].argmax(axis=0).cpu().detach().numpy()
        target_mask = target[index].cpu().detach().numpy()
        score += mean_foreground_dice(target_mask, predicted_mask)
    return score / prediction.shape[0]

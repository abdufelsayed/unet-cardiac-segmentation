"""Training and validation loops used by the final experiment."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import batch_foreground_dice


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return the sum of batch losses."""
    model.train()
    training_loss = 0.0

    for image, target in data_loader:
        image = image.unsqueeze(1).to(device=device)
        target = target.long().to(device=device)
        optimizer.zero_grad()
        prediction = model(image)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    return training_loss


def validate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model and return summed batch loss and batch Dice."""
    model.eval()
    score = 0.0
    validation_loss = 0.0

    with torch.no_grad():
        for image, target in data_loader:
            image = image.unsqueeze(1).to(device=device)
            target = target.long().to(device=device)
            prediction = model(image)
            score += batch_foreground_dice(prediction, target)
            validation_loss += criterion(prediction, target).item()

    return validation_loss, score

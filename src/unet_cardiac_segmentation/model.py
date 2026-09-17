"""U-Net architecture used for the cardiac segmentation experiment."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """Apply two padded convolutions, each followed by batch normalisation and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class Down(nn.Module):
    """Downsample by max pooling, then apply a double-convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class Up(nn.Module):
    """Upsample, concatenate the encoder features, and apply two convolutions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.block = DoubleConv(in_channels, out_channels)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        inputs = self.up(inputs)
        return self.block(torch.cat((skip, inputs), dim=1))


class OutConv(nn.Module):
    """Project the final feature map to one logit channel per class."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)


class UNet(nn.Module):
    """Four-level U-Net with one input channel and four output classes by default."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

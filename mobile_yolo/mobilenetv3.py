"""MobileNetV3 building blocks used by the lightweight YOLOv12 variants.

The public ``MobileNetV3Stage`` module intentionally preserves its channel
count.  This lets it be registered with Ultralytics without modifying the
installed package's ``parse_model`` implementation, while the YAML files can
still replace expensive backbone stages and downsampling convolutions.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def make_divisible(value: float, divisor: int = 8) -> int:
    """Round a channel count up to a hardware-friendly multiple."""
    return max(divisor, int(math.ceil(value / divisor) * divisor))


def activation_layer(name: str) -> nn.Module:
    """Create the MobileNetV3 activation requested by a YAML argument."""
    normalized = name.strip().lower()
    if normalized in {"hs", "hardswish", "hard_swish"}:
        return nn.Hardswish(inplace=True)
    if normalized in {"re", "relu"}:
        return nn.ReLU(inplace=True)
    if normalized in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBNActivation(nn.Sequential):
    """Convolution, batch normalization, and activation."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: str = "HS",
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(c1, c2, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            activation_layer(activation),
        )


class SqueezeExcitation(nn.Module):
    """MobileNetV3 squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, squeeze_ratio: float = 0.25) -> None:
        super().__init__()
        squeezed = make_divisible(channels * squeeze_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeezed, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeezed, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * scale


class InvertedResidual(nn.Module):
    """MobileNetV3 inverted residual with depthwise convolution and optional SE."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: float = 4.0,
        use_se: bool = True,
        activation: str = "HS",
    ) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        hidden = make_divisible(c1 * expansion)
        layers: list[nn.Module] = []
        if hidden != c1:
            layers.append(ConvBNActivation(c1, hidden, 1, activation=activation))
        layers.append(
            ConvBNActivation(
                hidden,
                hidden,
                kernel_size,
                stride,
                groups=hidden,
                activation=activation,
            )
        )
        if use_se:
            layers.append(SqueezeExcitation(hidden))
        layers.extend((nn.Conv2d(hidden, c2, 1, bias=False), nn.BatchNorm2d(c2)))

        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_residual else y


class MobileNetV3Stage(nn.Module):
    """One or more channel-preserving MobileNetV3 inverted residual blocks.

    Args mirror the values used in the model YAML files:
    ``channels, repeats, stride, expansion, kernel_size, use_se, activation``.
    Only the first block applies ``stride``; subsequent blocks use stride 1.
    """

    def __init__(
        self,
        channels: int,
        repeats: int = 1,
        stride: int = 1,
        expansion: float = 4.0,
        kernel_size: int = 3,
        use_se: bool = True,
        activation: str = "HS",
    ) -> None:
        super().__init__()
        if channels <= 0 or repeats <= 0:
            raise ValueError("channels and repeats must be positive")
        blocks = [
            InvertedResidual(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=stride if index == 0 else 1,
                expansion=expansion,
                use_se=use_se,
                activation=activation,
            )
            for index in range(repeats)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

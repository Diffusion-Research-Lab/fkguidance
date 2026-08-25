"""Positive time-conditioned reward models."""

import math

import torch


__all__ = ["PositiveRewardCNN", "PositiveRewardMLP"]


class _TimeFeatures(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        frequencies = torch.exp(torch.linspace(0, math.log(1000), width))
        self.register_buffer("frequencies", 2 * math.pi * frequencies, persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time.flatten().unsqueeze(1) * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class PositiveRewardMLP(torch.nn.Module):
    """Positive reward model for vector states."""

    def __init__(self, dim: int, data_scale: float | torch.Tensor = 1.0, hidden_dim: int = 128,
                 depth: int = 3) -> None:
        super().__init__()
        self.register_buffer("data_scale", torch.as_tensor(data_scale).float().clamp_min(1e-6))
        self.time_features = _TimeFeatures(hidden_dim // 2)
        layers = [torch.nn.Linear(dim + hidden_dim, hidden_dim), torch.nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU()))
        layers.append(torch.nn.Linear(hidden_dim, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        values = self.network(torch.cat((x.flatten(1) / self.data_scale, self.time_features(time)), dim=1)).flatten()
        return torch.nn.functional.softplus(values) + 1e-6


class _ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(math.gcd(in_channels, 8), in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm2 = torch.nn.GroupNorm(math.gcd(out_channels, 8), out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (torch.nn.Identity() if in_channels == out_channels and stride == 1
                     else torch.nn.Conv2d(in_channels, out_channels, 1, stride=stride))
        self.time = torch.nn.Linear(time_dim, out_channels)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        x = x + self.time(time)[:, :, None, None]
        return self.conv2(torch.nn.functional.silu(self.norm2(x))) + residual


class PositiveRewardCNN(torch.nn.Module):
    """Positive reward model for image or latent states."""

    def __init__(self, channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        time_dim = 4 * hidden_channels
        self.time_features = _TimeFeatures(hidden_channels)
        self.time_mlp = torch.nn.Sequential(torch.nn.Linear(2 * hidden_channels, time_dim), torch.nn.SiLU(),
                                            torch.nn.Linear(time_dim, time_dim))
        self.stem = torch.nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.blocks = torch.nn.ModuleList((_ResidualBlock(hidden_channels, hidden_channels, time_dim),
                                           _ResidualBlock(hidden_channels, 2 * hidden_channels, time_dim, 2),
                                           _ResidualBlock(2 * hidden_channels, 4 * hidden_channels, time_dim, 2)))
        self.head = torch.nn.Sequential(torch.nn.GroupNorm(8, 4 * hidden_channels), torch.nn.SiLU(),
                                        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                        torch.nn.Linear(4 * hidden_channels, 1))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time = self.time_mlp(self.time_features(time))
        x = self.stem(x)
        for block in self.blocks:
            x = block(x, time)
        return torch.nn.functional.softplus(self.head(x).flatten()) + 1e-6

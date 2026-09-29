"""Low-rank time-conditioned log-reward models."""

import math
import torch


__all__ = ["CNNEncoder", "LowRankLogReward", "LowRankLogRewardCNN", "LowRankLogRewardMLP", "MLPEncoder"]


class MLPEncoder(torch.nn.Module):
    """Encode vector states into one terminal feature and low-rank residual features."""

    def __init__(self, dim: int, output_dim: int, data_scale: float | torch.Tensor = 1.0,
                 hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("data_scale", torch.as_tensor(data_scale).float().clamp_min(1e-6))
        layers = [torch.nn.Linear(dim, hidden_dim), torch.nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU()))
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(1) / self.data_scale)


class CNNEncoder(torch.nn.Module):
    """Encode spatial states into one terminal feature and low-rank residual features."""

    def __init__(self, channels: int, output_dim: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden_channels, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(hidden_channels, 2 * hidden_channels, 3, stride=2, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(2 * hidden_channels, 4 * hidden_channels, 3, stride=2, padding=1), torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(4 * hidden_channels, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LowRankLogReward(torch.nn.Module):
    """Combine nonlinear state and time features through a low-rank expansion."""

    def __init__(self, encoder: torch.nn.Module, rank: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        if rank <= 0 or getattr(encoder, "output_dim", None) != rank + 1:
            raise ValueError("encoder output_dim must equal rank + 1")
        self.encoder = encoder
        self.rank = rank
        self.time_network = torch.nn.Sequential(torch.nn.Linear(1, hidden_dim), torch.nn.SiLU(),
                                                torch.nn.Linear(hidden_dim, rank + 1))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        spatial = self.encoder(x)
        time = time.flatten().to(spatial)
        temporal = self.time_network(time.unsqueeze(1))
        residual = (spatial[:, 1:] * temporal[:, 1:]).sum(1) / math.sqrt(self.rank)
        return (1 - time) * temporal[:, 0] + time * spatial[:, 0] + time * (1 - time) * residual


class LowRankLogRewardMLP(LowRankLogReward):
    def __init__(self, dim: int, data_scale: float | torch.Tensor = 1.0, hidden_dim: int = 128,
                 depth: int = 3, rank: int = 16) -> None:
        super().__init__(MLPEncoder(dim, rank + 1, data_scale, hidden_dim, depth), rank, hidden_dim)


class LowRankLogRewardCNN(LowRankLogReward):
    def __init__(self, channels: int, hidden_channels: int = 64, rank: int = 16) -> None:
        super().__init__(CNNEncoder(channels, rank + 1, hidden_channels), rank, hidden_channels)

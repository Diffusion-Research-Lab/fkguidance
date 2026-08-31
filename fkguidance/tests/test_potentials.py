import torch
from torch.utils.data import TensorDataset
from fkguidance import DensityRatioPotential


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def test_density_ratio_potential_is_clipped():
    potential = DensityRatioPotential(Identity(), 2, clip=0.2)
    with torch.no_grad():
        for parameter in potential.head.parameters():
            parameter.zero_()
        potential.head[2].bias.fill_(100)

    assert torch.allclose(potential(torch.zeros(3, 2)), torch.full((3,), 0.2))


def test_density_ratio_smoothing_is_symmetric_and_reproducible():
    dataset = TensorDataset(torch.zeros(8, 2), torch.tensor([0.0] * 4 + [1.0] * 4))

    potential = DensityRatioPotential(Identity(), 2, smoothing_std=0.1)
    first = potential._classification_dataset(dataset, torch.ones(2), seed=0)
    second = potential._classification_dataset(dataset, torch.ones(2), seed=0)
    features, targets = first.tensors

    assert torch.allclose(features, second.tensors[0])
    assert features[~targets.bool()].std() > 0
    assert features[targets.bool()].std() > 0

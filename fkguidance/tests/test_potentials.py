import torch
from torch.utils.data import TensorDataset
from fkguidance import DensityRatioPotential


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def test_relative_density_ratio_is_bounded():
    potential = DensityRatioPotential(Identity(), 2, relative_alpha=0.2)
    with torch.no_grad():
        for parameter in potential.head.parameters():
            parameter.zero_()
        potential.head[2].bias.fill_(100)

    assert torch.all(potential(torch.zeros(3, 2)) <= -torch.log(torch.tensor(0.2)))


def test_density_ratio_smoothing_is_symmetric_and_reproducible():
    dataset = TensorDataset(torch.zeros(8, 2), torch.tensor([0.0] * 4 + [1.0] * 4))

    for alpha in (None, 0.25):
        potential = DensityRatioPotential(Identity(), 2, relative_alpha=alpha, smoothing_std=0.1)
        first = potential._classification_dataset(dataset, torch.ones(2), seed=0)
        second = potential._classification_dataset(dataset, torch.ones(2), seed=0)
        features, targets = first.tensors

        assert torch.allclose(features, second.tensors[0])
        assert features[~targets.bool()].std() > 0
        assert features[targets.bool()].std() > 0

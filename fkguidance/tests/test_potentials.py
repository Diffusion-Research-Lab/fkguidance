import torch
from fkguidance import DensityRatioPotential, RadialSurvivalPotential, binary_datasets


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def test_radial_survival_potential_rewards_missing_tail():
    generated = torch.linspace(0, 1, 100).unsqueeze(1)
    reference = torch.linspace(0, 2, 100).unsqueeze(1)
    potential = RadialSurvivalPotential(clip=5)
    results = potential.fit(binary_datasets(reference, generated))

    assert results["n_reference"] == 80
    assert potential(torch.tensor([[1.5]])) > potential(torch.tensor([[0.5]]))


def test_relative_density_ratio_is_bounded():
    potential = DensityRatioPotential(Identity(), 2, relative_alpha=0.2)
    with torch.no_grad():
        for parameter in potential.head.parameters():
            parameter.zero_()
        potential.head[2].bias.fill_(100)

    assert torch.all(potential(torch.zeros(3, 2)) <= -torch.log(torch.tensor(0.2)))

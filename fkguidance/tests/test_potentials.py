import torch
from torch.utils.data import TensorDataset
from fkguidance import ConditionalPathPotential, ConfidenceRatioPotential, DensityRatioPotential


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


def test_confidence_ratio_potential_fits_and_restores_linear_ensemble():
    generated, reference = -torch.ones(10, 2), torch.ones(10, 2)
    dataset = TensorDataset(torch.cat((generated, reference)), torch.tensor([0.] * 10 + [1.] * 10))
    potential = ConfidenceRatioPotential(Identity(), 2, n_estimators=3, n_jobs=1)

    diagnostics = potential.fit((dataset, dataset, dataset))
    values = potential(torch.stack((generated[0], reference[0])))
    restored = ConfidenceRatioPotential(Identity(), 2, n_estimators=3)
    restored.load_state_dict(potential.state_dict())

    assert values[0] < 0 < values[1]
    assert torch.allclose(restored(torch.stack((generated[0], reference[0]))), values)
    assert set(diagnostics["test"]) == {
        "loss", "accuracy", "generated_accuracy", "reference_accuracy", "active_fraction", "mean_reliability"
    }

    with torch.no_grad():
        restored.head.weight.zero_()
        restored.head.bias.copy_(torch.tensor([-1.0, 0.5, 1.0]))
    assert 0 < restored(torch.zeros(1, 2)) < restored.head.bias.mean()


def test_conditional_path_potential_smoke():
    generator = torch.Generator().manual_seed(0)
    generated = torch.randn(96, 1, generator=generator) * 0.3 - 1.5
    reference = torch.randn(96, 1, generator=generator) * 0.3 + 1.5
    values = torch.cat((generated, reference))
    targets = torch.cat((torch.zeros(len(generated)), torch.ones(len(reference))))
    dataset = TensorDataset(values, targets)
    potential = ConditionalPathPotential(Identity(), 1, hidden_dim=32, ratio_batch_size=32)

    diagnostics = potential.fit(
        (dataset, dataset, dataset),
        training_kwargs={"n_epochs": 40, "batch_size": 32, "learning_rate": 3e-3},
        seed=0,
    )
    ratios = potential(torch.tensor([[-1.5], [1.5]], dtype=torch.float64))

    assert diagnostics["training"]["best_epoch"] >= 1
    assert ratios.shape == (2,)
    assert ratios.dtype == torch.float64
    assert torch.isfinite(ratios).all()
    assert ratios[0] < 0 < ratios[1]

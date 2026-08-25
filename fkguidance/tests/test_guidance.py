import torch

from fkguidance import PositiveRewardMLP, TerminalPotential, anchor_probabilities, binary_datasets, fit_guidance


class CoordinatePotential(TerminalPotential):
    def raw_potential(self, x):
        return x[:, 0]


def test_anchor_probabilities_mix_uniform_and_potential_bias():
    terminals = torch.tensor([[0.0], [1.0], [2.0]])
    probabilities = anchor_probabilities(CoordinatePotential(), terminals, beta=0.5, eta=2.0)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))
    assert torch.all(probabilities >= 0.5 / 3)
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_fit_guidance_uses_independent_continuations():
    generated = torch.linspace(-1, 1, 60).unsqueeze(1)
    reference = generated + 0.5
    terminal_datasets = binary_datasets(reference, generated)

    def forward_noise(terminals, times, context):
        return terminals + times[:, None]

    def continue_from(states, times, n_continuations, context):
        offsets = torch.linspace(0, 0.2, n_continuations)
        return states[:, None] + offsets[None, :, None]

    potential, model, results = fit_guidance(
        CoordinatePotential(),
        PositiveRewardMLP(1, hidden_dim=8, depth=1),
        terminal_datasets,
        generated,
        forward_noise,
        continue_from,
        n_states=30,
        n_continuations=3,
        training_kwargs={"n_epochs": 2, "batch_size": 8, "learning_rate": 1e-3},
        batch_size=10,
    )
    assert isinstance(potential, CoordinatePotential)
    assert results["reward"]["n_continuations"] == 3
    assert results["reward"]["test_loss"] >= 0
    assert model(torch.zeros(2, 1), torch.zeros(2)).shape == (2,)

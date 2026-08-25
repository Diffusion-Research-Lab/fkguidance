import torch
from fkguidance import PositiveRewardMLP, Potential, binary_datasets, fit_guidance, terminal_probabilities, tune_guidance_scale
from fkguidance.guidance import _h_dataset


class CoordinatePotential(Potential):
    def raw_potential(self, x):
        return x[:, 0]


def test_terminal_probabilities_mix_uniform_and_potential_bias():
    terminals = torch.tensor([[0.0], [1.0], [2.0]])
    probabilities = terminal_probabilities(CoordinatePotential(), terminals, beta=0.5, eta=2.0)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))
    assert torch.all(probabilities >= 0.5 / 3)
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_tune_guidance_scale_selects_a_metric_and_optionally_returns_trials():
    best_scale, best = tune_guidance_scale((0.5, 1.0, 2.0), lambda scale: {"error": abs(scale - 1)}, "error")
    _, trials = tune_guidance_scale((0.5, 1.0, 2.0), lambda scale: scale, lambda scale: scale,
                                    direction="max", return_trials=True)

    assert best_scale == 1.0
    assert best == {"error": 0.0}
    assert trials == {0.5: 0.5, 1.0: 1.0, 2.0: 2.0}


def test_h_dataset_includes_the_exact_terminal_condition():
    terminals = torch.linspace(-1, 1, 10).unsqueeze(1)

    def forward_noise(values, times, context):
        return values + times[:, None]

    def continue_from(states, times, n_continuations, context):
        return states[:, None].expand(-1, n_continuations, -1)

    dataset = _h_dataset(terminals, None, CoordinatePotential(), forward_noise, continue_from,
                         n_states=10, n_continuations=2, gamma=1.0, beta=0.5, eta=1.0,
                         time_group_size=3, device="cpu", seed=0, split="train")
    states, times, h_targets = dataset.tensors
    terminal = times == 1

    assert terminal.sum() == 2
    assert torch.allclose(h_targets[terminal], states[terminal, 0].exp())
    assert len(times.unique()) == 4


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
        time_group_size=10,
    )
    assert isinstance(potential, CoordinatePotential)
    assert results["reward"]["n_continuations"] == 3
    assert results["reward"]["test_loss"] >= 0
    assert model(torch.zeros(2, 1), torch.zeros(2)).shape == (2,)

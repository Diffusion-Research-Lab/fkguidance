import math
import torch
from fkguidance import LogRewardMLP, Potential, binary_datasets, fit_guidance, make_guidance, terminal_probabilities, tune_guidance_scale
from fkguidance.guidance import _log_h_dataset, _log_reward_loss


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


def test_tune_guidance_scale_uses_successive_budgets_and_keeps_the_first_round():
    calls = []

    def run(scale, budget):
        calls.append((scale, budget))
        return {"error": abs(scale - 2.0), "budget": budget}

    best_scale, trials = tune_guidance_scale((0.5, 1.0, 2.0, 4.0), run, "error",
                                             budgets=(10, 30, 100), return_trials=True)

    assert best_scale == 2.0
    assert len(calls) == 7
    assert set(trials) == {0.5, 1.0, 2.0, 4.0}
    assert {trial["budget"] for trial in trials.values()} == {10}


def test_tune_guidance_scale_refines_promising_log_scale_regions():
    def run(scale, budget):
        return {"error": abs(math.log(scale / 4)), "budget": budget}

    best_scale, rounds = tune_guidance_scale((1, 4, 16, 64), run, "error", budgets=(10, 30, 100),
                                             n_keep=(1, 2), refine=True, return_rounds=True)

    assert best_scale == 4
    assert [len(round_) for round_ in rounds] == [4, 3, 2]
    assert set(rounds[1]) == {2, 4, 8}


def test_log_h_dataset_includes_the_exact_terminal_condition():
    terminals = torch.linspace(-1, 1, 10).unsqueeze(1)

    def forward_noise(values, times, context):
        return values + times[:, None]

    def continue_from(states, times, n_continuations, context):
        return torch.stack((states, states + 2), dim=1)

    dataset = _log_h_dataset(terminals, None, CoordinatePotential(), forward_noise, continue_from,
                             n_states=10, n_continuations=2, gamma=1.0, beta=0.5, eta=1.0,
                             time_group_size=3, device="cpu", seed=0, split="train")
    states, times, log_h_targets = dataset.tensors
    terminal = times == 1

    assert terminal.sum() == 2
    expected = states[:, 0].clone()
    expected[~terminal] += torch.logsumexp(torch.tensor([0.0, 2.0]), dim=0) - math.log(2)
    assert torch.allclose(log_h_targets, expected)
    assert len(times.unique()) == 4


def test_log_reward_loss_targets_log_mean_exponential():
    targets = torch.tensor([-2.0, 0.0, 1.0], requires_grad=True)
    prediction = (torch.logsumexp(targets.detach(), dim=0) - math.log(len(targets))).requires_grad_()

    loss = _log_reward_loss(prediction.expand_as(targets), targets)
    prediction_gradient, target_gradient = torch.autograd.grad(loss, (prediction, targets), allow_unused=True)

    assert torch.allclose(prediction_gradient, torch.zeros_like(prediction_gradient), atol=1e-6)
    assert target_gradient is None
    assert not torch.isclose(prediction.detach(), targets.mean())


def test_make_guidance_differentiates_the_log_reward():
    class QuadraticLogReward(torch.nn.Module):
        def forward(self, x, time):
            return x.square().sum(dim=1)

    values = torch.tensor([[1.0, -2.0]])
    guidance = make_guidance(QuadraticLogReward(), scale=0.5)

    assert torch.allclose(guidance(values, torch.zeros(1)), values)


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
        LogRewardMLP(1, hidden_dim=8, depth=1),
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
    assert results["log_reward"]["objective"] == "exponential_log_reward"
    assert results["log_reward"]["n_continuations"] == 3
    assert results["log_reward"]["test_loss"] >= 0
    assert model(torch.zeros(2, 1), torch.zeros(2)).shape == (2,)

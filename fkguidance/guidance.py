"""Potential-guided continuation fitting."""

from collections.abc import Callable
import logging
import math
from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset

from .potentials import TerminalPotential
from .training import evaluate, select_parameters, train


__all__ = ["anchor_probabilities", "fit_guidance", "make_guidance"]


logger = logging.getLogger(__name__)


@torch.inference_mode()
def anchor_probabilities(potential: TerminalPotential, terminals: torch.Tensor, *, beta: float = 0.5,
                         eta: float = 1.0, batch_size: int = 1024,
                         device: str | torch.device = "cpu") -> torch.Tensor:
    """Mix uniform sampling with a softmax biased toward large terminal potentials."""
    if not 0 <= beta <= 1 or eta < 0:
        raise ValueError("beta must lie in [0, 1] and eta must be non-negative")
    potential.to(device).eval()
    values = torch.cat([potential(terminals[start:start + batch_size].to(device)).cpu()
                        for start in range(0, len(terminals), batch_size)])
    biased = torch.softmax(eta * values.double(), dim=0)
    return ((1 - beta) / len(values) + beta * biased).float()


@torch.inference_mode()
def _reward_dataset(terminals: torch.Tensor, context: torch.Tensor | None, potential: TerminalPotential,
                    forward_noise: Callable, continue_from: Callable, *, n_states: int, n_continuations: int,
                    gamma: float, beta: float, eta: float, batch_size: int, device: str | torch.device,
                    seed: int) -> TensorDataset:
    probabilities = anchor_probabilities(potential, terminals, beta=beta, eta=eta,
                                         batch_size=batch_size, device=device)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.multinomial(probabilities, n_states, replacement=True, generator=generator)
    selected_context = None if context is None else context[indices]
    states, times, rewards = [], [], []

    for start in range(0, n_states, batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_context = None if selected_context is None else selected_context[start:start + batch_size]
        # One continuously sampled time per batch keeps restart samplers vectorized without introducing a fixed grid.
        time = torch.full((len(batch_indices),), float(torch.rand((), generator=generator)))
        state = forward_noise(terminals[batch_indices], time, batch_context)
        continuations = continue_from(state, time, n_continuations, batch_context)
        shape = continuations.shape[:2]
        tau = potential(continuations.flatten(0, 1).to(device)).reshape(shape)
        reward = torch.exp(gamma * tau).mean(dim=1)
        states.append(state.cpu())
        times.append(time)
        rewards.append(reward.cpu())
        logger.info("continuation targets: %d/%d", min(start + batch_size, n_states), n_states)

    return TensorDataset(torch.cat(states), torch.cat(times), torch.cat(rewards))


def fit_guidance(potential: TerminalPotential, reward_model: torch.nn.Module,
                 terminal_datasets: tuple[Dataset, Dataset, Dataset], terminal_pool: torch.Tensor,
                 forward_noise: Callable, continue_from: Callable, *, context: torch.Tensor | None = None,
                 n_states: int, n_continuations: int, gamma: float = 1.0, beta: float = 0.5, eta: float = 1.0,
                 potential_kwargs: dict[str, Any] | None = None, training_kwargs: dict[str, Any],
                 pilot_kwargs: dict[str, Any] | None = None, batch_size: int = 256,
                 device: str | torch.device = "cpu", seed: int = 0) -> tuple[TerminalPotential, torch.nn.Module, dict]:
    """Fit tau, estimate h by stochastic continuations, and fit a positive reward model."""
    if min(len(terminal_pool), n_states, n_continuations, batch_size) <= 0:
        raise ValueError("pool and sample counts must be positive")
    if context is not None and len(context) != len(terminal_pool):
        raise ValueError("context and terminal pool must have equal length")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")

    logger.info("guidance stage 1/2: fitting the terminal potential")
    potential_results = potential.fit(terminal_datasets, device=device, seed=seed, **(potential_kwargs or {}))

    logger.info("guidance stage 2/2: sampling anchors and stochastic continuations")
    order = torch.randperm(len(terminal_pool), generator=torch.Generator().manual_seed(seed))
    bounds = (0, int(0.8 * len(order)), int(0.9 * len(order)), len(order))
    state_counts = (int(0.8 * n_states), int(0.1 * n_states), n_states - int(0.9 * n_states))
    datasets = []
    for split, (start, stop), count in zip(("train", "validation", "test"),
                                           zip(bounds[:-1], bounds[1:], strict=True), state_counts, strict=True):
        indices = order[start:stop]
        split_context = None if context is None else context[indices]
        dataset = _reward_dataset(terminal_pool[indices], split_context, potential, forward_noise, continue_from,
                                  n_states=count, n_continuations=n_continuations, gamma=gamma, beta=beta, eta=eta,
                                  batch_size=batch_size, device=device, seed=seed + len(datasets))
        datasets.append(dataset)
        logger.info("prepared %d %s reward observations", len(dataset), split)

    # A common scalar normalization improves conditioning and leaves grad log h unchanged.
    reward_scale = datasets[0].tensors[-1].mean().clamp_min(1e-8)
    datasets = [TensorDataset(dataset.tensors[0], dataset.tensors[1], dataset.tensors[2] / reward_scale)
                for dataset in datasets]
    loss_fn = torch.nn.functional.mse_loss
    selected, trials = {}, []
    if pilot_kwargs is not None:
        selected, trials = select_parameters(
            reward_model,
            datasets[0],
            datasets[1],
            list(pilot_kwargs["candidates"]),
            n_epochs=int(pilot_kwargs["n_epochs"]),
            loss_fn=loss_fn,
            device=device,
            seed=seed)
    parameters = {**training_kwargs, **selected}
    history = train(reward_model, datasets[0], datasets[1], loss_fn=loss_fn, device=device, seed=seed,
                    **parameters)
    final_batch_size = int(parameters.get("batch_size", 128))
    test_loss = evaluate(reward_model, datasets[2], final_batch_size, loss_fn, device)
    reward_model.cpu()
    return potential.cpu(), reward_model, {
        "potential": potential_results,
        "reward": {"gamma": gamma, "anchor_beta": beta, "anchor_eta": eta,
                   "n_states": n_states, "n_continuations": n_continuations,
                   "normalization": float(reward_scale), "selected": selected, "trials": trials,
                   "training": history, "test_loss": test_loss},
    }


def make_guidance(reward_model: torch.nn.Module, scale: float = 1.0) -> Callable:
    """Return scale times the spatial gradient of log h."""
    def guidance(x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            inputs = x.detach().requires_grad_(True)
            log_reward = reward_model(inputs, time).clamp_min(1e-8).log()
            return scale * torch.autograd.grad(log_reward.sum(), inputs)[0]

    return guidance

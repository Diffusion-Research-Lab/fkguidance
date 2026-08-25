"""Potential-guided continuation fitting."""

from collections.abc import Callable
import logging
import math
from typing import Any
import torch
from torch.utils.data import Dataset, TensorDataset
from .potentials import Potential
from .training import evaluate, select_parameters, train


__all__ = ["fit_guidance", "make_guidance", "terminal_probabilities", "tune_guidance_scale"]


logger = logging.getLogger(__name__)
_TERMINAL_FRACTION = 0.2


def _log_reward_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Proper loss whose population minimizer is log E[exp(target) | input]."""
    return (torch.exp(target.detach() - prediction) + prediction).mean()


@torch.inference_mode()
def terminal_probabilities(potential: Potential, terminals: torch.Tensor, *, beta: float = 0.5,
                           eta: float = 1.0, batch_size: int = 1024,
                           device: str | torch.device = "cpu") -> torch.Tensor:
    """Mix uniform terminal sampling with a softmax biased toward large tau values."""
    if not 0 <= beta <= 1 or eta < 0:
        raise ValueError("beta must lie in [0, 1] and eta must be non-negative")

    potential.to(device).eval()
    values = torch.cat([potential(terminals[start:start + batch_size].to(device)).cpu()
                        for start in range(0, len(terminals), batch_size)])

    # The uniform component preserves broad coverage; eta controls the bias toward deficient regions.
    biased = torch.softmax(eta * values.double(), dim=0)
    return ((1 - beta) / len(values) + beta * biased).float()


@torch.inference_mode()
def _log_h_dataset(terminals: torch.Tensor, context: torch.Tensor | None, potential: Potential,
                   forward_noise: Callable, continue_from: Callable, *, n_states: int, n_continuations: int,
                   gamma: float, beta: float, eta: float, time_group_size: int, device: str | torch.device,
                   seed: int, split: str) -> TensorDataset:
    """Build (X_t, t, log h) observations from terminal samples and stochastic continuations."""
    probabilities = terminal_probabilities(potential, terminals, beta=beta, eta=eta, device=device)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.multinomial(probabilities, n_states, replacement=True, generator=generator)

    # Exact terminal observations impose log h(x, 1) = gamma tau(x).
    n_terminal = max(1, int(_TERMINAL_FRACTION * n_states))
    terminal_indices, indices = indices[:n_terminal], indices[n_terminal:]
    terminal_states = terminals[terminal_indices]
    terminal_tau = potential(terminal_states.to(device)).cpu()
    states = [terminal_states.cpu()]
    times = [torch.ones(n_terminal)]
    log_h_targets = [gamma * terminal_tau]
    selected_context = None if context is None else context[indices]

    n_batches = math.ceil(len(indices) / time_group_size)
    n_reports = 4 if split == "train" else 1
    log_every = max(1, math.ceil(n_batches / n_reports))

    for batch_index, start in enumerate(range(0, len(indices), time_group_size), 1):
        batch_indices = indices[start:start + time_group_size]
        batch_context = None if selected_context is None else selected_context[start:start + time_group_size]

        # One continuously sampled time per batch keeps restart samplers vectorized without introducing a fixed grid.
        time = torch.full((len(batch_indices),), float(torch.rand((), generator=generator)))
        state = forward_noise(terminals[batch_indices], time, batch_context)
        continuations = continue_from(state, time, n_continuations, batch_context)

        # Stable Monte Carlo estimate of log h = log E[exp(gamma tau(X_1)) | X_t].
        shape = continuations.shape[:2]
        tau = potential(continuations.flatten(0, 1).to(device)).reshape(shape)
        log_h_target = torch.logsumexp(gamma * tau, dim=1) - math.log(n_continuations)

        states.append(state.cpu())
        times.append(time)
        log_h_targets.append(log_h_target.cpu())

        if batch_index % log_every == 0 or batch_index == n_batches:
            logger.info("log h targets | %s | %d/%d states", split,
                        min(n_terminal + start + time_group_size, n_states), n_states)

    return TensorDataset(torch.cat(states), torch.cat(times), torch.cat(log_h_targets))


def fit_guidance(potential: Potential, log_reward_model: torch.nn.Module,
                 terminal_datasets: tuple[Dataset, Dataset, Dataset], terminal_pool: torch.Tensor,
                 forward_noise: Callable, continue_from: Callable, *, context: torch.Tensor | None = None,
                 n_states: int, n_continuations: int, gamma: float = 1.0, beta: float = 0.5, eta: float = 1.0,
                 potential_kwargs: dict[str, Any] | None = None, training_kwargs: dict[str, Any],
                 pilot_kwargs: dict[str, Any] | None = None, time_group_size: int = 256,
                 device: str | torch.device = "cpu", seed: int = 0) -> tuple[Potential, torch.nn.Module, dict]:
    """Fit tau, construct Monte Carlo targets for log h, and fit u_phi = log h."""
    if min(len(terminal_pool), n_states, n_continuations, time_group_size) <= 0:
        raise ValueError("pool and sample counts must be positive")
    if context is not None and len(context) != len(terminal_pool):
        raise ValueError("context and terminal pool must have equal length")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")

    # tau is either analytic or learned from reference and generated terminal samples.
    logger.info("tau | fit | start | %s", type(potential).__name__, extra={"core_step": True})
    potential_results = potential.fit(terminal_datasets, device=device, seed=seed, **(potential_kwargs or {}))

    if "test" in potential_results:
        test = potential_results["test"]
        logger.info("tau | fit | done | accuracy=%.2f%% | generated=%.2f%% | reference=%.2f%%",
                    100 * test["accuracy"], 100 * test["generated_accuracy"], 100 * test["reference_accuracy"],
                    extra={"core_step": True})
    else:
        logger.info("tau | ready | %s", type(potential).__name__, extra={"core_step": True})

    # Keep terminal-pool splits disjoint before drawing terminal samples, states, and continuations.
    logger.info("log h targets | start | states=%d | time_group=%d | continuations/state=%d | "
                "gamma=%.3g | terminal_beta=%.3g | terminal_eta=%.3g", n_states, time_group_size,
                n_continuations, gamma, beta, eta, extra={"core_step": True})
    order = torch.randperm(len(terminal_pool), generator=torch.Generator().manual_seed(seed))
    pool_bounds = (0, int(0.8 * len(order)), int(0.9 * len(order)), len(order))
    state_counts = (int(0.8 * n_states), int(0.1 * n_states), n_states - int(0.9 * n_states))
    datasets = []

    split_ranges = zip(("train", "validation", "test"), pool_bounds[:-1], pool_bounds[1:], state_counts,
                       strict=True)
    for split_index, (split, start, stop, count) in enumerate(split_ranges):
        indices = order[start:stop]
        split_context = None if context is None else context[indices]
        dataset = _log_h_dataset(terminal_pool[indices], split_context, potential, forward_noise, continue_from,
                                 n_states=count, n_continuations=n_continuations, gamma=gamma, beta=beta, eta=eta,
                                 time_group_size=time_group_size, device=device, seed=seed + split_index, split=split)
        datasets.append(dataset)

    logger.info("log h targets | ready | train=%d | validation=%d | test=%d",
                *(len(dataset) for dataset in datasets), extra={"core_step": True})

    # An additive center improves conditioning and leaves grad log h unchanged.
    centering = datasets[0].tensors[-1].mean()
    datasets = [TensorDataset(dataset.tensors[0], dataset.tensors[1], dataset.tensors[2] - centering)
                for dataset in datasets]

    selected, trials = {}, []
    logger.info("log h | fit | start", extra={"core_step": True})

    if pilot_kwargs is not None:
        selected, trials = select_parameters(
            log_reward_model,
            datasets[0],
            datasets[1],
            list(pilot_kwargs["candidates"]),
            n_epochs=int(pilot_kwargs["n_epochs"]),
            loss_fn=_log_reward_loss,
            device=device,
            seed=seed,
            label="log h")

    parameters = {**training_kwargs, **selected}
    history = train(log_reward_model, datasets[0], datasets[1], loss_fn=_log_reward_loss, device=device, seed=seed,
                    label="log h", **parameters)
    final_batch_size = int(parameters.get("batch_size", 128))
    test_loss = evaluate(log_reward_model, datasets[2], final_batch_size, _log_reward_loss, device)

    logger.info("log h | fit | done | best_epoch=%d | validation_loss=%.6g | test_loss=%.6g",
                history["best_epoch"], min(history["validation_loss"]), test_loss, extra={"core_step": True})
    log_reward_model.cpu()

    return potential.cpu(), log_reward_model, {
        "potential": potential_results,
        "log_reward": {"objective": "exponential_log_reward", "gamma": gamma,
                       "terminal_beta": beta, "terminal_eta": eta,
                       "n_states": n_states, "n_continuations": n_continuations,
                       "terminal_fraction": _TERMINAL_FRACTION, "time_group_size": time_group_size,
                       "centering": float(centering), "selected": selected, "trials": trials,
                       "training": history, "test_loss": test_loss},
    }


def make_guidance(log_reward_model: torch.nn.Module, scale: float = 1.0) -> Callable:
    """Return the guidance vector V(x, t) = scale * grad_x u_phi(x, t)."""
    def guidance(x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            inputs = x.detach().requires_grad_(True)
            log_reward = log_reward_model(inputs, time)
            return scale * torch.autograd.grad(log_reward.sum(), inputs)[0]

    return guidance


def tune_guidance_scale(scales, run_scale: Callable[[float], Any], metric: str | Callable[[Any], float], *,
                        direction: str = "min", return_trials: bool = False) -> tuple[float, Any]:
    """Run a discrete scale search and return the best run or every trial."""
    scales = tuple(float(scale) for scale in scales)
    if not scales or len(scales) != len(set(scales)) or any(not math.isfinite(scale) for scale in scales):
        raise ValueError("scales must be a non-empty sequence of distinct finite values")
    if direction not in {"min", "max"}:
        raise ValueError("direction must be 'min' or 'max'")

    score = (lambda result: result[metric]) if isinstance(metric, str) else metric
    trials = {}
    best_scale, best_result, best_value = scales[0], None, None
    for scale in scales:
        result = run_scale(scale)
        value = float(score(result))
        if not math.isfinite(value):
            raise ValueError(f"metric is not finite for guidance scale {scale:g}")
        if return_trials:
            trials[scale] = result
        if best_value is None or (value < best_value if direction == "min" else value > best_value):
            best_scale, best_result, best_value = scale, result, value

    return best_scale, trials if return_trials else best_result

"""Potential-guided continuation fitting."""

from collections.abc import Callable
import logging
import math
from typing import Any
import torch
from torch.utils.data import Dataset, TensorDataset
from .potentials import DensityRatioPotential
from .training import evaluate, select_parameters, train


__all__ = ["fit_guidance", "make_guidance", "terminal_probabilities", "tune_guidance_scale"]


logger = logging.getLogger(__name__)


def _log_reward_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Proper loss whose population minimizer is log E[exp(target) | input]."""
    return (torch.exp(target.detach() - prediction) + prediction).mean()


def _target_diagnostics(dataset: TensorDataset) -> dict[str, float]:
    target = dataset.tensors[-1]
    return {"mean": float(target.mean()), "std": float(target.std(unbiased=False)),
            "null_loss": float(_log_reward_loss(torch.zeros_like(target), target))}


@torch.inference_mode()
def _terminal_diagnostics(log_reward_model: torch.nn.Module, potential: torch.nn.Module,
                          terminals: torch.Tensor, gamma: float, centering: torch.Tensor,
                          batch_size: int, device: str | torch.device) -> dict[str, float | int]:
    log_reward_model.to(device).eval()
    potential.to(device).eval()
    predictions, targets = [], []
    for start in range(0, len(terminals), batch_size):
        states = terminals[start:start + batch_size].to(device)
        time = states.new_ones(len(states))
        predictions.append(log_reward_model(states, time).cpu())
        targets.append((gamma * potential(states) - centering.to(device)).cpu())

    prediction, target = torch.cat(predictions), torch.cat(targets)
    error = prediction - target
    return {"samples": len(target), "target_std": float(target.std(unbiased=False)),
            "null_loss": float(_log_reward_loss(torch.zeros_like(target), target)),
            "loss": float(_log_reward_loss(prediction, target)),
            "rmse": float(error.square().mean().sqrt()), "mae": float(error.abs().mean())}


@torch.inference_mode()
def _terminal_weights(potential: DensityRatioPotential, terminals: torch.Tensor, eta: float, beta: float,
                      batch_size: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not math.isfinite(eta) or eta < 0 or not 0 <= beta <= 1:
        raise ValueError("eta must be non-negative and beta must lie in [0, 1]")

    potential.to(device).eval()
    values = torch.cat([potential(terminals[start:start + batch_size].to(device)).cpu()
                        for start in range(0, len(terminals), batch_size)])

    tilted = torch.softmax(eta * values.double(), dim=0)
    proposal = (1 - beta) / len(values) + beta * tilted
    return proposal.float(), tilted


def terminal_probabilities(potential: DensityRatioPotential, terminals: torch.Tensor, *, eta: float = 1.0,
                           beta: float = 1.0, batch_size: int = 1024,
                           device: str | torch.device = "cpu") -> torch.Tensor:
    """Sample from (1 - beta) p_theta + beta p_theta exp(eta tau) / Z."""
    return _terminal_weights(potential, terminals, eta, beta, batch_size, device)[0]


@torch.no_grad()
def _log_h_dataset(terminals: torch.Tensor, context: torch.Tensor | None, potential: DensityRatioPotential,
                   forward_noise: Callable, continue_from: Callable, *, n_states: int, n_continuations: int,
                   gamma: float, eta: float, beta: float, time_group_size: int, terminal_fraction: float = 0.25,
                   device: str | torch.device, seed: int,
                   split: str) -> tuple[TensorDataset, dict[str, float | int | str]]:
    """Build (X_t, t, log h) observations from terminal samples and stochastic continuations."""
    probabilities, tilted = _terminal_weights(potential, terminals, eta, beta, 1024, device)
    ess = float(tilted.square().sum().reciprocal())
    selection = {"split": split, "pool_size": len(terminals), "eta": eta, "beta": beta,
                 "ess": ess, "ess_fraction": ess / len(terminals)}
    logger.info("log h terminal weights | %s | eta=%.3g | beta=%.3g | ESS=%.1f/%d (%.2f%%)",
                split, eta, beta, ess, len(terminals), 100 * selection["ess_fraction"])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.multinomial(probabilities, n_states, replacement=True, generator=generator)
    selected_context = None if context is None else context[indices]
    states, times, log_h_targets = [], [], []

    n_terminal = round(terminal_fraction * n_states)
    if n_terminal:
        terminal_states = terminals[indices[:n_terminal]].to(device)
        states.append(terminal_states.cpu())
        times.append(torch.ones(n_terminal))
        log_h_targets.append((gamma * potential(terminal_states)).cpu())

    indices = indices[n_terminal:]
    selected_context = None if selected_context is None else selected_context[n_terminal:]
    n_batches = math.ceil(len(indices) / time_group_size)
    quantiles = (torch.arange(n_batches) + torch.rand(n_batches, generator=generator)) / max(1, n_batches)
    batch_times = torch.sin(math.pi * quantiles / 2)
    batch_times = batch_times[torch.randperm(n_batches, generator=generator)]
    n_reports = 4 if split == "train" else 1
    log_every = max(1, math.ceil(n_batches / n_reports))

    for batch_index, start in enumerate(range(0, len(indices), time_group_size), 1):
        batch_indices = indices[start:start + time_group_size]
        batch_context = None if selected_context is None else selected_context[start:start + time_group_size]

        # The stratified half-cosine law emphasizes the difficult near-terminal region.
        time_value = batch_times[batch_index - 1].item()
        time = torch.full((len(batch_indices),), time_value)
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

    selection["terminal_fraction"] = n_terminal / n_states
    return TensorDataset(torch.cat(states), torch.cat(times), torch.cat(log_h_targets)), selection


def fit_guidance(potential: DensityRatioPotential, log_reward_model: torch.nn.Module,
                 terminal_datasets: tuple[Dataset, Dataset, Dataset], terminal_pool: torch.Tensor,
                 forward_noise: Callable, continue_from: Callable, *, context: torch.Tensor | None = None,
                 n_states: int, n_continuations: int, gamma: float = 1.0, eta: float | None = None,
                 beta: float = 1.0, terminal_fraction: float = 0.25,
                 potential_kwargs: dict[str, Any] | None = None, training_kwargs: dict[str, Any],
                 pilot_kwargs: dict[str, Any] | None = None, time_group_size: int = 256,
                 device: str | torch.device = "cpu", seed: int = 0) -> tuple[DensityRatioPotential, torch.nn.Module, dict]:
    """Fit tau, construct Monte Carlo targets for log h, and fit v_phi = log h."""
    if min(len(terminal_pool), n_states, n_continuations, time_group_size) <= 0:
        raise ValueError("pool and sample counts must be positive")
    if context is not None and len(context) != len(terminal_pool):
        raise ValueError("context and terminal pool must have equal length")
    eta = gamma if eta is None else eta
    if not (math.isfinite(gamma) and gamma > 0 and math.isfinite(eta) and eta >= 0 and 0 <= beta <= 1 and math.isfinite(terminal_fraction) and 0 <= terminal_fraction <= 1):
        raise ValueError("gamma must be positive; eta non-negative; beta and terminal_fraction must lie in [0, 1]")

    # tau is either analytic or learned from reference and generated terminal samples.
    logger.info("tau | fit | start | %s", type(potential).__name__, extra={"core_step": True})
    potential_results = potential.fit(terminal_datasets, device=device, seed=seed, **(potential_kwargs or {}))

    if "test" in potential_results:
        test = potential_results["test"]
        if "training" in potential_results:
            history = potential_results["training"]
            best_epoch = history["best_epoch"]
            logger.info("tau | fit | done | best_epoch=%d | validation_loss=%.6g | test_loss=%.6g | "
                        "accuracy=%.2f%% | generated=%.2f%% | reference=%.2f%%",
                        best_epoch, history["validation_loss"][best_epoch - 1], test["loss"],
                        100 * test["accuracy"], 100 * test["generated_accuracy"],
                        100 * test["reference_accuracy"], extra={"core_step": True})
        else:
            logger.info("tau | fit | done | estimators=%d | confidence=%.1f%% | test_loss=%.6g | "
                        "accuracy=%.2f%% | generated=%.2f%% | reference=%.2f%% | active=%.2f%% | reliability=%.2f%%",
                        potential_results["n_estimators"], 100 * potential_results["confidence"], test["loss"],
                        100 * test["accuracy"], 100 * test["generated_accuracy"],
                        100 * test["reference_accuracy"], 100 * test["active_fraction"],
                        100 * test["mean_reliability"],
                        extra={"core_step": True})
    else:
        logger.info("tau | ready | %s", type(potential).__name__, extra={"core_step": True})

    # Keep terminal-pool splits disjoint before drawing terminal samples, states, and continuations.
    logger.info("log h targets | start | states=%d | terminal=%.1f%% | time_group=%d | continuations/state=%d | "
                "gamma=%.3g | eta=%.3g | beta=%.3g", n_states, 100 * terminal_fraction, time_group_size,
                n_continuations, gamma, eta, beta,
                extra={"core_step": True})
    order = torch.randperm(len(terminal_pool), generator=torch.Generator().manual_seed(seed))
    pool_bounds = (0, int(0.8 * len(order)), int(0.9 * len(order)), len(order))
    state_counts = (int(0.8 * n_states), int(0.1 * n_states), n_states - int(0.9 * n_states))
    datasets, selections = [], []

    split_ranges = zip(("train", "validation", "test"), pool_bounds[:-1], pool_bounds[1:], state_counts,
                       strict=True)
    for split_index, (split, start, stop, count) in enumerate(split_ranges):
        indices = order[start:stop]
        split_context = None if context is None else context[indices]
        dataset, selection = _log_h_dataset(
            terminal_pool[indices], split_context, potential, forward_noise, continue_from,
            n_states=count, n_continuations=n_continuations, gamma=gamma, eta=eta, beta=beta,
            time_group_size=time_group_size, terminal_fraction=terminal_fraction,
            device=device, seed=seed + split_index, split=split)
        datasets.append(dataset)
        selections.append(selection)

    logger.info("log h targets | ready | train=%d | validation=%d | test=%d",
                *(len(dataset) for dataset in datasets), extra={"core_step": True})

    # An additive center improves conditioning and leaves grad log h unchanged.
    centering = datasets[0].tensors[-1].mean()
    datasets = [TensorDataset(dataset.tensors[0], dataset.tensors[1], dataset.tensors[2] - centering)
                for dataset in datasets]
    target_diagnostics = {
        split: _target_diagnostics(dataset)
        for split, dataset in zip(("train", "validation", "test"), datasets, strict=True)
    }
    for split, diagnostics in target_diagnostics.items():
        logger.info("log h targets | %s | mean=%.6g | std=%.6g | null_loss=%.6g",
                    split, diagnostics["mean"], diagnostics["std"], diagnostics["null_loss"])

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
    terminal_indices = order[pool_bounds[-2]:pool_bounds[-1]]
    terminal_diagnostics = _terminal_diagnostics(
        log_reward_model, potential, terminal_pool[terminal_indices], gamma, centering, final_batch_size, device)

    logger.info("log h | fit | done | best_epoch=%d | validation_loss=%.6g | test_loss=%.6g",
                history["best_epoch"], min(history["validation_loss"]), test_loss, extra={"core_step": True})
    logger.info("log h terminal anchor | test | samples=%d | target_std=%.6g | null_loss=%.6g | "
                "loss=%.6g | rmse=%.6g | mae=%.6g",
                terminal_diagnostics["samples"], terminal_diagnostics["target_std"],
                terminal_diagnostics["null_loss"], terminal_diagnostics["loss"],
                terminal_diagnostics["rmse"], terminal_diagnostics["mae"], extra={"core_step": True})
    log_reward_model.cpu()

    return potential.cpu(), log_reward_model, {
        "potential": potential_results,
        "log_reward": {"terminal_selection": selections, "selected": selected, "trials": trials,
                       "training": history, "test_loss": test_loss, "target_diagnostics": target_diagnostics,
                       "terminal_diagnostics": terminal_diagnostics},
    }


def make_guidance(log_reward_model: torch.nn.Module, scale: float = 1.0) -> Callable:
    """Return the guidance vector V(x, t) = scale * grad_x v_phi(x, t)."""
    def guidance(x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            inputs = x.detach().requires_grad_(True)
            log_reward = log_reward_model(inputs, time)
            return scale * torch.autograd.grad(log_reward.sum(), inputs)[0]

    return guidance


def tune_guidance_scale(scales, run_scale: Callable[..., Any], metric: str | Callable[[Any], float], *,
                        direction: str = "min", budgets=None, n_keep=None, refine: bool = False,
                        return_trials: bool = False, return_rounds: bool = False) -> tuple[float, Any]:
    """Select a scale with optional successive budgets and one log-scale refinement."""
    scales = tuple(float(scale) for scale in scales)
    if not scales or len(scales) != len(set(scales)) or any(not math.isfinite(scale) for scale in scales):
        raise ValueError("scales must be a non-empty sequence of distinct finite values")
    if direction not in {"min", "max"}:
        raise ValueError("direction must be 'min' or 'max'")
    if budgets is not None:
        budgets = tuple(int(budget) for budget in budgets)
        if not budgets or any(budget <= 0 for budget in budgets) or any(a >= b for a, b in zip(budgets, budgets[1:])):
            raise ValueError("budgets must be a non-empty increasing sequence of positive integers")
    if n_keep is not None:
        n_keep = tuple(int(value) for value in n_keep)
        if budgets is None or len(n_keep) != len(budgets) - 1 or any(value <= 0 for value in n_keep):
            raise ValueError("n_keep must contain one positive value per non-final budget")
    if refine and (budgets is None or len(budgets) < 2 or any(scale <= 0 for scale in scales)):
        raise ValueError("refinement requires positive scales and at least two budgets")
    if return_trials and return_rounds:
        raise ValueError("return_trials and return_rounds are mutually exclusive")

    score = (lambda result: result[metric]) if isinstance(metric, str) else metric
    candidates, first_trials, rounds = scales, None, []
    for round_index, budget in enumerate((None,) if budgets is None else budgets):
        trials = {scale: run_scale(scale) if budget is None else run_scale(scale, budget) for scale in candidates}
        rounds.append(trials)
        values = {scale: float(score(result)) for scale, result in trials.items()}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("guidance-scale metric must be finite")
        first_trials = trials if first_trials is None else first_trials
        reverse = direction == "max"
        ranked = sorted(candidates, key=values.__getitem__, reverse=reverse)
        if budget is None or round_index == len(budgets) - 1:
            candidates = (ranked[0],)
            continue

        keep = n_keep[round_index] if n_keep is not None else max(1, len(ranked) // 2)
        survivors = tuple(ranked[:min(keep, len(ranked))])
        if refine and round_index == 0:
            ordered = sorted(candidates)
            refined = set(survivors)
            for scale in survivors:
                index = ordered.index(scale)
                if index:
                    refined.add(math.sqrt(ordered[index - 1] * scale))
                if index + 1 < len(ordered):
                    refined.add(math.sqrt(scale * ordered[index + 1]))
            candidates = tuple(sorted(refined))
        else:
            candidates = survivors

    best_scale = candidates[0]
    if return_rounds:
        return best_scale, rounds
    return best_scale, first_trials if return_trials else trials[best_scale]

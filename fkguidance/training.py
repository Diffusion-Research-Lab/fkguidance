"""Small supervised-training utilities used by F-K guidance."""

from collections.abc import Callable
import logging
import math
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset


__all__ = ["evaluate", "select_parameters", "train"]


logger = logging.getLogger(__name__)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, dataset: Dataset, batch_size: int, loss_fn: Callable,
             device: str | torch.device) -> float:
    model.to(device).eval()
    total = 0.0
    for *inputs, target in DataLoader(dataset, batch_size=batch_size):
        inputs = [value.to(device) for value in inputs]
        target = target.to(device)
        total += float(loss_fn(model(*inputs), target)) * len(target)
    return total / len(dataset)


def train(model: torch.nn.Module, train_dataset: Dataset, validation_dataset: Dataset, *, n_epochs: int = 100,
          batch_size: int = 128, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
          learning_rate_half_life: float = 32.0, loss_fn: Callable = torch.nn.functional.mse_loss,
          device: str | torch.device = "cpu", seed: int = 0, label: str = "model") -> dict[str, Any]:
    """Train a model and restore its best validation state."""
    if min(n_epochs, batch_size) <= 0 or min(len(train_dataset), len(validation_dataset)) <= 0:
        raise ValueError("epochs, batch size, and dataset sizes must be positive")

    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5**(1 / learning_rate_half_life))
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history = {"train_loss": [], "validation_loss": [], "gradient_norm": []}
    best_loss, best_epoch, best_state = math.inf, 0, None
    log_every = max(5, math.ceil(n_epochs / 8))

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        for *inputs, target in loader:
            inputs = [value.to(device) for value in inputs]
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(*inputs), target)
            loss.backward()
            gradients = [parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None]
            total_grad_norm += float(torch.stack(gradients).norm())
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)

        train_loss = total_loss / len(train_dataset)
        validation_loss = evaluate(model, validation_dataset, batch_size, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["gradient_norm"].append(total_grad_norm / len(loader))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if (epoch + 1) % log_every == 0 or epoch + 1 == n_epochs:
            logger.info("%s fit | epoch %d/%d | train_loss=%.6g | validation_loss=%.6g | grad_norm=%.6g | lr=%.3g",
                        label, epoch + 1, n_epochs, train_loss, validation_loss, history["gradient_norm"][-1],
                        optimizer.param_groups[0]["lr"])
        scheduler.step()

    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, **history}


def select_parameters(model: torch.nn.Module, train_dataset: Dataset, validation_dataset: Dataset,
                      candidates: list[dict[str, Any]], *, n_epochs: int, loss_fn: Callable,
                      device: str | torch.device, seed: int,
                      label: str = "model") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select training parameters from identical initial weights."""
    if not candidates:
        return {}, []

    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    trials = []
    for index, parameters in enumerate(candidates, 1):
        model.load_state_dict(initial_state)
        logger.info("%s pilot | candidate %d/%d | %s", label, index, len(candidates), parameters)
        history = train(model, train_dataset, validation_dataset, n_epochs=n_epochs, loss_fn=loss_fn,
                        device=device, seed=seed, label=f"{label} pilot", **parameters)
        epoch = history["best_epoch"] - 1
        trials.append({"parameters": parameters,
                       "best_epoch": history["best_epoch"],
                       "train_loss": history["train_loss"][epoch],
                       "validation_loss": history["validation_loss"][epoch],
                       "history": history})

    model.load_state_dict(initial_state)
    selected = min(trials, key=lambda trial: trial["validation_loss"])
    logger.info("%s pilot | selected %s | validation_loss=%.6g",
                label, selected["parameters"], selected["validation_loss"])
    return dict(selected["parameters"]), trials

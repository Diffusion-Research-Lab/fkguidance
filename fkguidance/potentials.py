"""Terminal potentials defining a target distribution correction."""

from abc import ABC, abstractmethod
import math
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from .training import evaluate, select_parameters, train


__all__ = ["DensityRatioPotential", "Potential"]


class Potential(torch.nn.Module, ABC):
    """A scalar terminal potential tau used to define the F-K log reward."""

    def __init__(self, clip: float = 10.0) -> None:
        super().__init__()
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive")
        self.clip = float(clip)

    def fit(self, datasets: tuple[Dataset, Dataset, Dataset], **kwargs) -> dict[str, Any]:
        """Fit data-dependent state, if any."""
        return {"name": type(self).__name__}

    @abstractmethod
    def raw_potential(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.raw_potential(x)
        if values.shape != (len(x),):
            raise ValueError(f"potential shape {tuple(values.shape)} does not match {(len(x),)}")
        return values.clamp(-self.clip, self.clip)


class DensityRatioPotential(Potential):
    """Estimate a log density ratio in a supplied embedding with a balanced classifier.

    Reference samples have label one. With equal class priors, the classifier logit estimates
    log(p_reference / p_generated), or its relative-density-ratio counterpart when alpha is given.
    Optional Gaussian feature noise estimates the corresponding ratio of smoothed distributions.
    """

    def __init__(self, embedding: torch.nn.Module, output_dim: int, relative_alpha: float | None = None,
                 smoothing_std: float = 0.0, hidden_dim: int = 128, **kwargs) -> None:
        super().__init__(**kwargs)
        if relative_alpha is not None and not 0 < relative_alpha < 1:
            raise ValueError("relative_alpha must lie in (0, 1)")
        if not math.isfinite(smoothing_std) or smoothing_std < 0:
            raise ValueError("smoothing_std must be finite and non-negative")
        self.embedding = embedding
        self.relative_alpha = relative_alpha
        self.smoothing_std = float(smoothing_std)
        self.head = torch.nn.Sequential(torch.nn.Linear(output_dim, hidden_dim), torch.nn.SiLU(),
                                        torch.nn.Linear(hidden_dim, 1), torch.nn.Flatten(0))

    @torch.no_grad()
    def _embed(self, dataset: Dataset, batch_size: int, device: str | torch.device) -> TensorDataset:
        """Compute frozen features once before fitting the small classifier head."""
        self.embedding.to(device).eval()
        features, targets = [], []

        for inputs, target in DataLoader(dataset, batch_size=batch_size):
            features.append(self.embedding(inputs.to(device)).float().cpu())
            targets.append(target.float())

        return TensorDataset(torch.cat(features), torch.cat(targets))

    def _classification_dataset(self, dataset: TensorDataset, feature_scale: torch.Tensor,
                                seed: int) -> TensorDataset:
        """Build a balanced reference-versus-generated-or-mixture dataset."""
        features, targets = dataset.tensors
        reference, generated = features[targets.bool()], features[~targets.bool()]
        n_samples = min(len(reference), len(generated))
        generator = torch.Generator().manual_seed(seed)

        reference = reference[torch.randperm(len(reference), generator=generator)[:n_samples]]
        generated = generated[torch.randperm(len(generated), generator=generator)[:n_samples]]

        if self.relative_alpha is None:
            negative = generated
        else:
            n_reference = round(self.relative_alpha * n_samples)
            negative = torch.cat((reference[:n_reference], generated[:n_samples - n_reference]))
            negative = negative[torch.randperm(n_samples, generator=generator)]

        features = torch.cat((negative, reference))
        if self.smoothing_std:
            noise = torch.randn(features.shape, generator=generator, dtype=features.dtype)
            features = features + self.smoothing_std * feature_scale * noise

        return TensorDataset(features,
                             torch.cat((torch.zeros(n_samples), torch.ones(n_samples))))

    def fit(self, datasets: tuple[Dataset, Dataset, Dataset], *, training_kwargs: dict[str, Any],
            pilot_kwargs: dict[str, Any] | None = None, embedding_batch_size: int = 256,
            device: str | torch.device = "cpu", seed: int = 0) -> dict[str, Any]:
        """Embed the three splits, fit the ratio classifier, and report its test behavior."""
        embedded = tuple(self._embed(dataset, embedding_batch_size, device) for dataset in datasets)
        feature_scale = embedded[0].tensors[0].std(dim=0).clamp_min(1e-6)
        train_dataset, validation_dataset, test_dataset = tuple(
            self._classification_dataset(dataset, feature_scale, seed + index)
            for index, dataset in enumerate(embedded))

        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        selected, trials = {}, []

        if pilot_kwargs is not None:
            selected, trials = select_parameters(
                self.head,
                train_dataset,
                validation_dataset,
                list(pilot_kwargs["candidates"]),
                n_epochs=int(pilot_kwargs["n_epochs"]),
                loss_fn=loss_fn,
                device=device,
                seed=seed,
                label="tau")

        parameters = {**training_kwargs, **selected}
        history = train(self.head, train_dataset, validation_dataset, loss_fn=loss_fn, device=device, seed=seed,
                        label="tau", **parameters)
        batch_size = int(parameters.get("batch_size", 128))
        test_loss = evaluate(self.head, test_dataset, batch_size, loss_fn, device)

        # Audit the fitted ratio on the original generated-versus-reference test split.
        logits, targets = [], []
        self.head.eval()
        with torch.inference_mode():
            for features, target in DataLoader(embedded[2], batch_size=batch_size):
                logits.append(self.head(features.to(device)).cpu())
                targets.append(target.bool())

        logits, targets = torch.cat(logits), torch.cat(targets)
        correct = (logits >= 0) == targets
        self.cpu()

        return {"name": type(self).__name__, "relative_alpha": self.relative_alpha,
                "smoothing_std": self.smoothing_std, "selected": selected,
                "trials": trials, "training": history,
                "test": {"loss": test_loss, "accuracy": float(correct.float().mean()),
                         "generated_accuracy": float(correct[~targets].float().mean()),
                         "reference_accuracy": float(correct[targets].float().mean())}}

    def raw_potential(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.embedding(x))
        if self.relative_alpha is not None:
            # The exact relative ratio is at most 1 / alpha; enforce that bound on its learned estimate.
            logits = logits.clamp_max(-math.log(self.relative_alpha))
        return logits

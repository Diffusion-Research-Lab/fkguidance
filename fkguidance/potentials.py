"""Terminal potential estimated from reference and generated samples."""

from contextlib import redirect_stdout
from functools import cache
import importlib
import io
import logging
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from joblib import Parallel, delayed
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.linear_model import LogisticRegression
from .training import evaluate, select_parameters, train


__all__ = ["DensityRatioPotential", "ConfidenceRatioPotential", "ConditionalPathPotential"]


logger = logging.getLogger(__name__)


@cache
def _conditional_path_api() -> tuple[Any, ...]:
    """Load the pinned upstream implementation without making it a package dependency."""
    root = Path(__file__).resolve().parents[1] / "vendor" / "dre_prob_paths"
    if not (root / "prob_path_lib.py").is_file():
        raise ImportError(
            "ConditionalPathPotential requires the dre-prob-paths submodule; run "
            "`git submodule update --init vendor/dre_prob_paths` from the fkguidance repository"
        )

    # density_ratios imports the authors' image dataset module although its toy evaluator does not use it.
    # A temporary stub avoids pulling JAX and torchvision into this otherwise small adapter.
    previous_datasets = sys.modules.get("datasets")
    datasets_stub = ModuleType("datasets")
    datasets_stub.logit_transform = None
    sys.modules["datasets"] = datasets_stub
    sys.path.insert(0, str(root))
    try:
        paths = importlib.import_module("prob_path_lib")
        losses = importlib.import_module("toy_losses")
        networks = importlib.import_module("models.toy_networks")
        ratios = importlib.import_module("density_ratios")
    except Exception as error:
        raise ImportError(
            "Could not import the vendored dre-prob-paths CTSM-v implementation; "
            "install fkguidance dependencies and initialize the submodule"
        ) from error
    finally:
        sys.path.remove(str(root))
        if previous_datasets is None:
            del sys.modules["datasets"]
        else:
            sys.modules["datasets"] = previous_datasets

    return (paths.TwoSB, networks.TimeScoreNetwork, losses.get_optimizer,
            losses.toy_optimization_manager, losses.get_step_fn, ratios.get_toy_density_ratio_fn)


@torch.no_grad()
def _embed(embedding: torch.nn.Module, dataset: Dataset, batch_size: int,
           device: str | torch.device) -> TensorDataset:
    """Compute frozen features once before fitting a classifier."""
    embedding.to(device).eval()
    features, targets = [], []
    for inputs, target in DataLoader(dataset, batch_size=batch_size):
        features.append(embedding(inputs.to(device)).float().cpu())
        targets.append(target.cpu())
    return TensorDataset(torch.cat(features), torch.cat(targets))


class DensityRatioPotential(torch.nn.Module):
    """Estimate a log density ratio in a supplied embedding with a balanced classifier.

    Reference samples have label one. With equal class priors, the classifier logit estimates
    log(p_reference / p_generated). Optional Gaussian feature noise estimates the ratio between
    smoothed distributions.
    """

    def __init__(self, embedding: torch.nn.Module, output_dim: int, smoothing_std: float = 0.0,
                 hidden_dim: int = 128, clip: float = 10.0) -> None:
        super().__init__()
        if not math.isfinite(smoothing_std) or smoothing_std < 0:
            raise ValueError("smoothing_std must be finite and non-negative")
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive")

        self.embedding = embedding
        self.smoothing_std = float(smoothing_std)
        self.clip = float(clip)
        self.head = torch.nn.Sequential(torch.nn.Linear(output_dim, hidden_dim),
                                        torch.nn.SiLU(),
                                        torch.nn.Linear(hidden_dim, 1),
                                        torch.nn.Flatten(0))

    def _classification_dataset(self, dataset: TensorDataset, feature_scale: torch.Tensor,
                                seed: int) -> TensorDataset:
        """Build a balanced generated-versus-reference feature dataset."""
        features, targets = dataset.tensors
        reference, generated = features[targets.bool()], features[~targets.bool()]
        n_samples = min(len(reference), len(generated))
        generator = torch.Generator().manual_seed(seed)

        reference = reference[torch.randperm(len(reference), generator=generator)[:n_samples]]
        generated = generated[torch.randperm(len(generated), generator=generator)[:n_samples]]
        features = torch.cat((generated, reference))
        if self.smoothing_std:
            noise = torch.randn(features.shape, generator=generator, dtype=features.dtype)
            features = features + self.smoothing_std * feature_scale * noise

        return TensorDataset(features,
                             torch.cat((torch.zeros(n_samples), torch.ones(n_samples))))

    def fit(self, datasets: tuple[Dataset, Dataset, Dataset], *, training_kwargs: dict[str, Any],
            pilot_kwargs: dict[str, Any] | None = None, embedding_batch_size: int = 256,
            device: str | torch.device = "cpu", seed: int = 0) -> dict[str, Any]:
        """Embed the three splits, fit the ratio classifier, and report its test behavior."""
        embedded = tuple(_embed(self.embedding, dataset, embedding_batch_size, device) for dataset in datasets)
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

        return {"name": type(self).__name__, "smoothing_std": self.smoothing_std, "selected": selected,
                "trials": trials, "training": history,
                "test": {"loss": test_loss, "accuracy": float(correct.float().mean()),
                         "generated_accuracy": float(correct[~targets].float().mean()),
                         "reference_accuracy": float(correct[targets].float().mean())}}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the clipped terminal potential tau(x)."""
        return self.head(self.embedding(x)).clamp(-self.clip, self.clip)


class ConfidenceRatioPotential(torch.nn.Module):
    """Conservative log-density ratio from bootstrapped logistic regressions."""

    def __init__(self, embedding: torch.nn.Module, output_dim: int, n_estimators: int = 20,
                 confidence: float = 0.9,
                 smoothing_std: float = 0.0, C: float = 1.0, clip: float = 10.0,
                 n_jobs: int = -1) -> None:
        super().__init__()
        if output_dim <= 0 or n_estimators < 2:
            raise ValueError("output_dim must be positive and n_estimators must be at least two")
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        if not math.isfinite(smoothing_std) or smoothing_std < 0:
            raise ValueError("smoothing_std must be finite and non-negative")
        if not math.isfinite(C) or C <= 0 or not math.isfinite(clip) or clip <= 0:
            raise ValueError("C and clip must be finite and positive")

        self.embedding = embedding
        self.confidence = float(confidence)
        self.smoothing_std = float(smoothing_std)
        self.C = float(C)
        self.clip = float(clip)
        self.n_jobs = int(n_jobs)
        self.head = torch.nn.Linear(output_dim, n_estimators).requires_grad_(False)

    def _fit_one(self, reference: np.ndarray, generated: np.ndarray, scale: np.ndarray,
                 seed: int) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed)
        n = min(len(reference), len(generated))
        x = np.concatenate((generated[rng.integers(len(generated), size=n)],
                            reference[rng.integers(len(reference), size=n)]))
        y = np.concatenate((np.zeros(n), np.ones(n)))
        if self.smoothing_std:
            x = x + self.smoothing_std * scale * rng.standard_normal(x.shape)
        model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000).fit(x, y)
        return model.coef_[0], float(model.intercept_[0])

    def fit(self, datasets: tuple[Dataset, Dataset, Dataset], *, embedding_batch_size: int = 256,
            device: str | torch.device = "cpu", seed: int = 0, **_: Any) -> dict[str, Any]:
        train, test = (_embed(self.embedding, datasets[index], embedding_batch_size, device) for index in (0, 2))

        x, y = train.tensors
        y = y.bool()
        reference, generated = x[y].numpy(), x[~y].numpy()
        scale = x.std(dim=0).clamp_min(1e-6).numpy()
        parameters = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one)(reference, generated, scale, seed + i)
            for i in range(self.head.out_features)
        )
        weights, biases = zip(*parameters, strict=True)
        with torch.no_grad():
            self.head.weight.copy_(torch.as_tensor(np.stack(weights)))
            self.head.bias.copy_(torch.as_tensor(biases))

        x_test, y_test = test.tensors
        y_test = y_test.bool()
        logits = self.head(x_test)
        mean_logits = logits.mean(dim=1)
        test_loss = torch.nn.functional.binary_cross_entropy_with_logits(mean_logits, y_test.float())
        correct = (mean_logits >= 0) == y_test
        lower, upper = self._bounds(logits)
        active = (lower > 0) | (upper < 0)
        _, reliability = self._estimate(logits)
        self.cpu()
        return {"name": type(self).__name__, "n_estimators": self.head.out_features,
                "confidence": self.confidence,
                "test": {"loss": float(test_loss), "accuracy": float(correct.float().mean()),
                         "generated_accuracy": float(correct[~y_test].float().mean()),
                         "reference_accuracy": float(correct[y_test].float().mean()),
                         "active_fraction": float(active.float().mean()),
                         "mean_reliability": float(reliability.mean())}}

    def _bounds(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = (1.0 - self.confidence) / 2.0
        return tuple(torch.quantile(logits, torch.tensor([alpha, 1 - alpha], device=logits.device), dim=1))

    def _estimate(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        center = logits.mean(dim=1)
        lower, upper = self._bounds(logits)
        uncertainty = torch.maximum(center - lower, upper - center)
        reliability = center.square() / (center.square() + uncertainty.square() + torch.finfo(center.dtype).eps)
        return center * reliability, reliability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau, _ = self._estimate(self.head(self.embedding(x).float()))
        return tau.to(x.dtype).clamp(-self.clip, self.clip)


class ConditionalPathPotential(torch.nn.Module):
    """Estimate log(p_reference / p_generated) with the authors' TwoSB CTSM-v method."""

    def __init__(self, embedding: torch.nn.Module, output_dim: int, path_std: float = 1.0,
                 hidden_dim: int = 256, clip: float = 10.0, ratio_batch_size: int = 256,
                 rtol: float = 1e-6, atol: float = 1e-6) -> None:
        super().__init__()
        if output_dim <= 0 or hidden_dim <= 0 or ratio_batch_size <= 0:
            raise ValueError("output_dim, hidden_dim, and ratio_batch_size must be positive")
        if not math.isfinite(path_std) or path_std <= 0:
            raise ValueError("path_std must be finite and positive")
        if not math.isfinite(clip) or clip <= 0 or any(not math.isfinite(value) or value <= 0
                                                       for value in (rtol, atol)):
            raise ValueError("clip, rtol, and atol must be finite and positive")

        TwoSB, TimeScoreNetwork, *_, ratio_factory = _conditional_path_api()
        config = SimpleNamespace(data=SimpleNamespace(dim=output_dim),
                                 model=SimpleNamespace(z_dim=hidden_dim))
        self.embedding = embedding
        self.head = TimeScoreNetwork(config)
        self.path = TwoSB(output_dim, var=path_std ** 2)
        self.path_std = float(path_std)
        self.clip = float(clip)
        self.ratio_batch_size = int(ratio_batch_size)
        self._ratio_fn = ratio_factory(rtol=rtol, atol=atol, eps1=1e-5, eps2=1e-5)

    @staticmethod
    def _endpoints(dataset: TensorDataset) -> tuple[torch.Tensor, torch.Tensor]:
        features, targets = dataset.tensors
        generated, reference = features[~targets.bool()], features[targets.bool()]
        if min(len(generated), len(reference)) == 0:
            raise ValueError("each split must contain generated and reference samples")
        return generated, reference

    @staticmethod
    def _sample(endpoints: tuple[torch.Tensor, torch.Tensor], batch_size: int,
                generator: torch.Generator, device: str | torch.device) -> list[torch.Tensor]:
        return [values[torch.randint(len(values), (batch_size,), generator=generator)].to(device)
                for values in endpoints]

    def _train(self, train_data: tuple[torch.Tensor, torch.Tensor],
               validation_data: tuple[torch.Tensor, torch.Tensor], *, n_epochs: int = 100,
               batch_size: int = 128, learning_rate: float = 1e-4, weight_decay: float = 0.0,
               device: str | torch.device = "cpu", seed: int = 0, label: str = "tau") -> dict[str, Any]:
        """Run the released optimizer and CTSM-v step with local data iteration/checkpointing."""
        if min(n_epochs, batch_size) <= 0:
            raise ValueError("n_epochs and batch_size must be positive")

        _, _, get_optimizer, optimization_manager, get_step_fn, _ = _conditional_path_api()
        optim = SimpleNamespace(optimizer="Adam", lr=learning_rate, beta1=0.9, eps=1e-8,
                                weight_decay=weight_decay, amsgrad=False, warmup=0, grad_clip=-1.0)
        config = SimpleNamespace(optim=optim)
        self.head.to(device)
        optimizer = get_optimizer(config, self.head.parameters())
        state = {"model": self.head, "optimizer": optimizer, "step": 0}
        common = dict(sde=None, eps1=1e-5, eps2=1e-5, eps_factor=1 - 2e-5, joint=False,
                      dsm=False, reweight="obj_var", conditional=True, prob_path=self.path,
                      factor=1.0, device=torch.device(device), batch_size=batch_size, full=True)
        train_step = get_step_fn(train=True, optimize_fn=optimization_manager(config), **common)
        validation_step = get_step_fn(train=False, **common)
        train_generator = torch.Generator().manual_seed(seed)
        validation_generator = torch.Generator().manual_seed(seed + 1)
        train_steps = max(1, math.ceil(min(map(len, train_data)) / batch_size))
        validation_steps = max(1, math.ceil(min(map(len, validation_data)) / batch_size))
        history = {"train_loss": [], "validation_loss": []}
        best_loss, best_epoch, best_state = math.inf, 0, None
        log_every = max(1, math.ceil(n_epochs / 8))
        torch.manual_seed(seed)

        for epoch in range(1, n_epochs + 1):
            train_loss = np.mean([train_step(state, self._sample(train_data, batch_size,
                                                                 train_generator, device))["loss"]
                                  for _ in range(train_steps)])
            validation_loss = np.mean([
                validation_step(state, self._sample(validation_data, batch_size,
                                                    validation_generator, device))["loss"]
                for _ in range(validation_steps)])
            history["train_loss"].append(float(train_loss))
            history["validation_loss"].append(float(validation_loss))
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in self.head.state_dict().items()}
            if epoch % log_every == 0 or epoch == n_epochs:
                logger.info("%s fit | epoch %d/%d | train_loss=%.6g | validation_loss=%.6g",
                            label, epoch, n_epochs, train_loss, validation_loss)

        self.head.load_state_dict(best_state)
        return {"best_epoch": best_epoch, **history}

    def _ratios(self, features: torch.Tensor) -> torch.Tensor:
        values = []
        for batch in features.split(self.ratio_batch_size):
            with redirect_stdout(io.StringIO()):
                ratio, _ = self._ratio_fn(self.head, batch, score_type="time")
            values.append(torch.as_tensor(ratio, device=features.device, dtype=features.dtype))
        return torch.cat(values)

    def fit(self, datasets: tuple[Dataset, Dataset, Dataset], *, training_kwargs: dict[str, Any],
            pilot_kwargs: dict[str, Any] | None = None, embedding_batch_size: int = 256,
            device: str | torch.device = "cpu", seed: int = 0) -> dict[str, Any]:
        """Embed both endpoints and fit the official TwoSB CTSM-v estimator."""
        embedded = tuple(_embed(self.embedding, dataset, embedding_batch_size, device) for dataset in datasets)
        endpoints = tuple(self._endpoints(data) for data in embedded)
        initial_state = {name: value.detach().cpu().clone() for name, value in self.head.state_dict().items()}
        selected, trials = {}, []

        if pilot_kwargs is not None:
            candidates = list(pilot_kwargs["candidates"])
            for index, parameters in enumerate(candidates, 1):
                self.head.load_state_dict(initial_state)
                logger.info("tau pilot | candidate %d/%d | %s", index,
                            len(candidates), parameters)
                history = self._train(endpoints[0], endpoints[1], n_epochs=int(pilot_kwargs["n_epochs"]),
                                      device=device, seed=seed, label="tau pilot", **parameters)
                epoch = history["best_epoch"] - 1
                trials.append({"parameters": parameters, "best_epoch": history["best_epoch"],
                               "train_loss": history["train_loss"][epoch],
                               "validation_loss": history["validation_loss"][epoch], "history": history})
            if trials:
                selected_trial = min(trials, key=lambda trial: trial["validation_loss"])
                selected = dict(selected_trial["parameters"])
                logger.info("tau pilot | selected %s | validation_loss=%.6g",
                            selected, selected_trial["validation_loss"])

        self.head.load_state_dict(initial_state)
        parameters = {**training_kwargs, **selected}
        history = self._train(endpoints[0], endpoints[1], device=device, seed=seed, **parameters)
        test_history = self._train_loss(endpoints[2], int(parameters.get("batch_size", 128)), device, seed + 2)
        self.to(device).eval()
        features, targets = embedded[2].tensors
        ratios = self._ratios(features.to(device)).cpu()
        targets = targets.bool()
        correct = (ratios >= 0) == targets
        self.cpu()
        return {"name": type(self).__name__, "path_std": self.path_std, "selected": selected,
                "trials": trials, "training": history,
                "test": {"loss": test_history, "accuracy": float(correct.float().mean()),
                         "generated_accuracy": float(correct[~targets].float().mean()),
                         "reference_accuracy": float(correct[targets].float().mean())}}

    def _train_loss(self, endpoints: tuple[torch.Tensor, torch.Tensor], batch_size: int,
                    device: str | torch.device, seed: int) -> float:
        """Evaluate the official stochastic CTSM-v objective."""
        _, _, _, _, get_step_fn, _ = _conditional_path_api()
        step = get_step_fn(sde=None, train=False, eps1=1e-5, eps2=1e-5, eps_factor=1 - 2e-5,
                           joint=False, dsm=False, reweight="obj_var", conditional=True,
                           prob_path=self.path, factor=1.0, device=torch.device(device),
                           batch_size=batch_size, full=True)
        state = {"model": self.head, "step": 0}
        generator = torch.Generator().manual_seed(seed)
        n_steps = max(1, math.ceil(min(map(len, endpoints)) / batch_size))
        return float(np.mean([step(state, self._sample(endpoints, batch_size, generator, device))["loss"]
                              for _ in range(n_steps)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate tau(x) = log p_reference(x) - log p_generated(x)."""
        dtype = x.dtype
        features = self.embedding(x).to(next(self.head.parameters()).dtype)
        ratios = self._ratios(features)
        return ratios.to(dtype).clamp(-self.clip, self.clip)

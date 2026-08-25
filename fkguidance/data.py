"""Tensor-dataset construction for terminal-potential fitting."""

import torch
from torch.utils.data import TensorDataset


__all__ = ["binary_datasets", "split_tensors"]


def split_tensors(*values: torch.Tensor, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> tuple[TensorDataset, ...]:
    if len(values) == 0 or any(len(value) != len(values[0]) for value in values):
        raise ValueError("values must be non-empty tensors of equal length")
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6 or min(fractions) <= 0:
        raise ValueError("fractions must contain three positive values summing to one")

    ends = (0, int(fractions[0] * len(values[0])), int(sum(fractions[:2]) * len(values[0])), len(values[0]))
    return tuple(TensorDataset(*(value[start:stop] for value in values))
                 for start, stop in zip(ends[:-1], ends[1:], strict=True))


def binary_datasets(reference: torch.Tensor, generated: torch.Tensor, seed: int = 0) -> tuple[TensorDataset, ...]:
    """Build balanced generated-versus-reference train, validation, and test datasets."""
    n_samples = min(len(reference), len(generated))
    generator = torch.Generator().manual_seed(seed)
    reference = reference[torch.randperm(len(reference), generator=generator)[:n_samples]]
    generated = generated[torch.randperm(len(generated), generator=generator)[:n_samples]]
    reference_splits = split_tensors(reference)
    generated_splits = split_tensors(generated)

    datasets = []
    for reference_split, generated_split in zip(reference_splits, generated_splits, strict=True):
        reference_values = reference_split.tensors[0]
        generated_values = generated_split.tensors[0]
        values = torch.cat((generated_values, reference_values))
        targets = torch.cat((torch.zeros(len(generated_values)), torch.ones(len(reference_values))))
        order = torch.randperm(len(values), generator=generator)
        datasets.append(TensorDataset(values[order], targets[order]))
    return tuple(datasets)

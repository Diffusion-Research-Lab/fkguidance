"""Post-training Feynman--Kac guidance."""

from .data import binary_datasets, split_tensors
from .guidance import fit_guidance, make_guidance, terminal_probabilities, tune_guidance_scale
from .models import LogRewardCNN, LogRewardMLP
from .potentials import DensityRatioPotential


__all__ = [
    "DensityRatioPotential",
    "LogRewardCNN",
    "LogRewardMLP",
    "binary_datasets",
    "fit_guidance",
    "make_guidance",
    "split_tensors",
    "terminal_probabilities",
    "tune_guidance_scale",
]

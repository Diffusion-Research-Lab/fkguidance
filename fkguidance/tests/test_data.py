import pytest
import torch

from fkguidance import binary_datasets, split_tensors


def test_binary_datasets_are_balanced_and_split_80_10_10():
    datasets = binary_datasets(torch.ones(10, 2), torch.zeros(10, 2), seed=2)

    assert [len(dataset) for dataset in datasets] == [16, 2, 2]
    assert all(dataset.tensors[1].sum() == len(dataset) / 2 for dataset in datasets)


def test_split_tensors_rejects_invalid_fractions():
    with pytest.raises(ValueError, match="fractions"):
        split_tensors(torch.zeros(10), fractions=(0.5, 0.4, 0.2))

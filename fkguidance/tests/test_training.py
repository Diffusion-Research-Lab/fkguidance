import torch
from torch.utils.data import TensorDataset

from fkguidance.training import train


def test_train_stops_after_stale_validation_epochs():
    dataset = TensorDataset(torch.ones(8, 1), torch.zeros(8, 1))
    history = train(torch.nn.Linear(1, 1), dataset, dataset, n_epochs=10, batch_size=4,
                    learning_rate=0., patience=2)

    assert history["best_epoch"] == 1
    assert len(history["validation_loss"]) == 3

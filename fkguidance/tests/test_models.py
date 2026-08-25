import torch
from fkguidance import PositiveRewardCNN, PositiveRewardMLP


def test_reward_models_are_positive_and_differentiable():
    models_and_inputs = ((PositiveRewardMLP(3, hidden_dim=8, depth=2), torch.randn(4, 3)),
                         (PositiveRewardCNN(2, hidden_channels=8), torch.randn(4, 2, 8, 8)))
    for model, values in models_and_inputs:
        values.requires_grad_()
        reward = model(values, torch.rand(4))
        reward.log().sum().backward()

        assert reward.shape == (4,)
        assert torch.all(reward > 0)
        assert values.grad is not None

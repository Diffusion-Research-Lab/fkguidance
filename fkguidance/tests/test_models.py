import torch
from fkguidance import LogRewardCNN, LogRewardMLP


def test_log_reward_models_are_scalar_and_differentiable():
    models_and_inputs = ((LogRewardMLP(3, hidden_dim=8, depth=2), torch.randn(4, 3)),
                         (LogRewardCNN(2, hidden_channels=8), torch.randn(4, 2, 8, 8)))
    for model, values in models_and_inputs:
        values.requires_grad_()
        log_reward = model(values, torch.rand(4))
        log_reward.sum().backward()

        assert log_reward.shape == (4,)
        assert values.grad is not None

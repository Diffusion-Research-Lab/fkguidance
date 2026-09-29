import torch
from fkguidance import (CNNEncoder, LowRankLogReward, LowRankLogRewardCNN,
                        LowRankLogRewardMLP, MLPEncoder)


def test_low_rank_log_reward_models_are_scalar_and_differentiable():
    models_and_inputs = ((LowRankLogRewardMLP(3, hidden_dim=8, depth=2, rank=4), torch.randn(4, 3)),
                         (LowRankLogRewardCNN(2, hidden_channels=8, rank=4), torch.randn(4, 2, 8, 8)))
    for model, values in models_and_inputs:
        values.requires_grad_()
        log_reward = model(values, torch.rand(4))
        log_reward.sum().backward()

        assert log_reward.shape == (4,)
        assert values.grad is not None


def test_low_rank_log_reward_endpoints():
    model = LowRankLogReward(MLPEncoder(3, 5, hidden_dim=8, depth=1), rank=4, hidden_dim=8)
    values = torch.randn(4, 3, requires_grad=True)

    noisy = model(values, torch.zeros(4))
    noisy_gradient = torch.autograd.grad(noisy.sum(), values, retain_graph=True)[0]
    terminal = model(values, torch.ones(4))

    assert torch.count_nonzero(noisy_gradient) == 0
    assert torch.equal(terminal, model.encoder(values)[:, 0])


def test_low_rank_wrappers_use_matching_encoders():
    assert isinstance(LowRankLogRewardMLP(3).encoder, MLPEncoder)
    assert isinstance(LowRankLogRewardCNN(2).encoder, CNNEncoder)

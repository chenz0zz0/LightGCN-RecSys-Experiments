import torch


def test_bpr_softplus_matches_logsigmoid_form():
    pos = torch.tensor([1.2, -0.1, 0.8])
    neg = torch.tensor([0.2, 0.3, -0.4])
    a = torch.nn.functional.softplus(neg - pos).mean()
    b = -torch.nn.functional.logsigmoid(pos - neg).mean()
    assert torch.allclose(a, b, atol=1e-7)


def test_l2_sum_matches_norm_squared():
    x = torch.randn(32, 8)
    assert torch.allclose(x.pow(2).sum(), x.norm(2).pow(2), atol=1e-5)


if __name__ == '__main__':
    test_bpr_softplus_matches_logsigmoid_form()
    test_l2_sum_matches_norm_squared()
    print('LightGCN math audit: PASS')

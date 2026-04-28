from __future__ import annotations

import torch
import pytest

from dtcalib.deep_learning.train import (
    build_model,
    gaussian_nll_loss,
    compute_batch_loss_and_pred,
)


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Model factory
# ============================================================

@pytest.mark.parametrize(
    "model_name,expected_mode,expected_class,input_channels,input_shape,output_dim",
    [
        ("cnn", "deterministic", "RCInverseCNN", 6, (4, 6, 128), 3),
        ("prob_cnn", "probabilistic", "ProbabilisticRCInverseCNN", 6, (4, 6, 128), 3),
        ("mlp", "deterministic", "RCInverseMLP", 3, (4, 24), 3),
    ],
)
def test_build_model_forward_shapes(
    model_name: str,
    expected_mode: str,
    expected_class: str,
    input_channels: int,
    input_shape: tuple[int, ...],
    output_dim: int,
) -> None:
    model, mode, class_name = build_model(
        model_name,
        output_dim=output_dim,
        input_channels=input_channels,
    )

    assert mode == expected_mode
    assert class_name == expected_class
    assert _count_trainable_params(model) > 0

    x = torch.randn(*input_shape)

    out = model(x)

    if expected_mode == "deterministic":
        assert isinstance(out, torch.Tensor)
        assert out.shape == (input_shape[0], output_dim)
        assert torch.all(torch.isfinite(out))

    else:
        mu, log_var = out
        assert mu.shape == (input_shape[0], output_dim)
        assert log_var.shape == (input_shape[0], output_dim)
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(log_var))


def test_build_model_accepts_uppercase_name() -> None:
    model, mode, class_name = build_model("CNN", output_dim=3, input_channels=6)

    assert mode == "deterministic"
    assert class_name == "RCInverseCNN"

    x = torch.randn(2, 6, 64)
    y = model(x)

    assert y.shape == (2, 3)


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown_model", output_dim=3, input_channels=6)


# ============================================================
# Gaussian NLL
# ============================================================

def test_gaussian_nll_loss_matches_manual_formula() -> None:
    mu = torch.tensor([[0.0, 1.0]])
    target = torch.tensor([[1.0, 3.0]])
    log_var = torch.zeros_like(mu)

    loss = gaussian_nll_loss(mu, log_var, target)

    manual = 0.5 * torch.mean((target - mu) ** 2)

    assert loss.item() == pytest.approx(manual.item())


def test_gaussian_nll_loss_is_scalar_finite_and_differentiable() -> None:
    mu = torch.zeros(8, 3, requires_grad=True)
    log_var = torch.zeros(8, 3, requires_grad=True)
    target = torch.ones(8, 3)

    loss = gaussian_nll_loss(mu, log_var, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert mu.grad is not None
    assert log_var.grad is not None
    assert torch.all(torch.isfinite(mu.grad))
    assert torch.all(torch.isfinite(log_var.grad))


def test_gaussian_nll_loss_penalizes_overconfident_wrong_prediction() -> None:
    target = torch.ones(16, 3)
    mu_wrong = torch.zeros(16, 3)

    loss_small_var = gaussian_nll_loss(
        mu_wrong,
        torch.full_like(mu_wrong, -10.0),
        target,
    )
    loss_large_var = gaussian_nll_loss(
        mu_wrong,
        torch.full_like(mu_wrong, 2.0),
        target,
    )

    assert loss_small_var > loss_large_var


# ============================================================
# compute_batch_loss_and_pred
# ============================================================

def test_compute_batch_loss_deterministic_cnn() -> None:
    model, mode, _ = build_model("cnn", output_dim=3, input_channels=6)

    x = torch.randn(4, 6, 128)
    y = torch.randn(4, 3)

    loss, pred, pred_std = compute_batch_loss_and_pred(model, x, y, mode)

    assert loss.ndim == 0
    assert pred.shape == (4, 3)
    assert pred_std is None
    assert torch.isfinite(loss)


def test_compute_batch_loss_probabilistic_cnn() -> None:
    model, mode, _ = build_model("prob_cnn", output_dim=3, input_channels=6)

    x = torch.randn(4, 6, 128)
    y = torch.randn(4, 3)

    loss, pred, pred_std = compute_batch_loss_and_pred(model, x, y, mode)

    assert loss.ndim == 0
    assert pred.shape == (4, 3)
    assert pred_std is not None
    assert pred_std.shape == (4, 3)
    assert torch.all(pred_std > 0)
    assert torch.isfinite(loss)


def test_compute_batch_loss_deterministic_mlp() -> None:
    model, mode, _ = build_model("mlp", output_dim=3, input_channels=3)

    x = torch.randn(4, 24)
    y = torch.randn(4, 3)

    loss, pred, pred_std = compute_batch_loss_and_pred(model, x, y, mode)

    assert loss.ndim == 0
    assert pred.shape == (4, 3)
    assert pred_std is None
    assert torch.isfinite(loss)


def test_compute_batch_loss_invalid_mode_raises() -> None:
    model, _, _ = build_model("cnn", output_dim=3, input_channels=6)

    x = torch.randn(4, 6, 128)
    y = torch.randn(4, 3)

    with pytest.raises(ValueError, match="Unsupported model_mode"):
        compute_batch_loss_and_pred(model, x, y, "invalid_mode")


# ============================================================
# Training-step behaviour
# ============================================================

@pytest.mark.parametrize("model_name,input_channels,input_shape", [
    ("cnn", 6, (8, 6, 128)),
    ("prob_cnn", 6, (8, 6, 128)),
    ("mlp", 3, (8, 24)),
])
def test_one_training_step_updates_at_least_one_parameter(
    model_name: str,
    input_channels: int,
    input_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(0)

    model, mode, _ = build_model(
        model_name,
        output_dim=3,
        input_channels=input_channels,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(*input_shape)
    y = torch.randn(input_shape[0], 3)

    before = [
        p.detach().clone()
        for p in model.parameters()
        if p.requires_grad
    ]

    loss, _, _ = compute_batch_loss_and_pred(model, x, y, mode)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = [
        p.detach().clone()
        for p in model.parameters()
        if p.requires_grad
    ]

    assert torch.isfinite(loss)
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


@pytest.mark.parametrize("model_name,input_channels,input_shape", [
    ("cnn", 6, (8, 6, 128)),
    ("prob_cnn", 6, (8, 6, 128)),
])
def test_mini_train_eval_loop_runs_without_nan(
    model_name: str,
    input_channels: int,
    input_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(42)

    model, mode, _ = build_model(
        model_name,
        output_dim=3,
        input_channels=input_channels,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_train = torch.randn(*input_shape)
    y_train = torch.randn(input_shape[0], 3)

    x_val = torch.randn(*input_shape)
    y_val = torch.randn(input_shape[0], 3)

    train_losses = []

    model.train()
    for _ in range(3):
        loss, _, _ = compute_batch_loss_and_pred(model, x_train, y_train, mode)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        val_loss, pred, pred_std = compute_batch_loss_and_pred(model, x_val, y_val, mode)

    assert all(np_loss == pytest.approx(np_loss) for np_loss in train_losses)
    assert torch.isfinite(val_loss)
    assert pred.shape == (input_shape[0], 3)

    if mode == "probabilistic":
        assert pred_std is not None
        assert torch.all(pred_std > 0)
    else:
        assert pred_std is None


def test_cnn_supports_time_fft_six_channels() -> None:
    model, mode, _ = build_model("cnn", output_dim=3, input_channels=6)

    x = torch.randn(2, 6, 256)
    y = torch.randn(2, 3)

    loss, pred, pred_std = compute_batch_loss_and_pred(model, x, y, mode)

    assert pred.shape == (2, 3)
    assert pred_std is None
    assert torch.isfinite(loss)


def test_prob_cnn_supports_time_fft_six_channels() -> None:
    model, mode, _ = build_model("prob_cnn", output_dim=3, input_channels=6)

    x = torch.randn(2, 6, 256)
    y = torch.randn(2, 3)

    loss, pred, pred_std = compute_batch_loss_and_pred(model, x, y, mode)

    assert pred.shape == (2, 3)
    assert pred_std is not None
    assert pred_std.shape == (2, 3)
    assert torch.isfinite(loss)
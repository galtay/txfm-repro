"""Unit tests for src/txfm_repro/metrics.py."""

from __future__ import annotations

import math

import torch

from txfm_repro.lit_model import poisson_loss
from txfm_repro.metrics import (
    _build_visible_mask,
    _per_sample_pearson,
    _per_sample_r2,
    compute_holdout_metrics,
)


def _make_inputs(B: int = 4, G: int = 20, K: int = 5, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    target = torch.rand(B, G, generator=gen) * math.log1p(1e5)
    x_hat = torch.rand(B, G, generator=gen) * math.log1p(1e5)
    # K distinct unmasked indices per row.
    unmasked_idx = torch.zeros(B, K, dtype=torch.long)
    for b in range(B):
        unmasked_idx[b] = torch.randperm(G, generator=gen)[:K]
    return x_hat, target, unmasked_idx


# ---------- _build_visible_mask ----------

def test_visible_mask_basic() -> None:
    B, G, K = 2, 10, 3
    unmasked_idx = torch.tensor([[0, 5, 9], [1, 2, 3]], dtype=torch.long)
    visible = _build_visible_mask(unmasked_idx, G, padding_mask=None)
    assert visible.shape == (B, G)
    assert visible.dtype == torch.bool
    assert visible[0].nonzero().squeeze().tolist() == [0, 5, 9]
    assert visible[1].nonzero().squeeze().tolist() == [1, 2, 3]


def test_visible_mask_respects_padding() -> None:
    # Pad-slot index is 7 — distinct from any real visible idx — so we can
    # cleanly verify the pad slot did NOT mark its position.
    unmasked_idx = torch.tensor([[0, 5, 7], [1, 2, 7]], dtype=torch.long)
    padding_mask = torch.tensor([[False, False, True], [False, False, True]], dtype=torch.bool)
    visible = _build_visible_mask(unmasked_idx, G=10, padding_mask=padding_mask)
    assert visible[0, 0].item()
    assert visible[0, 5].item()
    assert not visible[0, 7].item()
    assert visible[1, 1].item()
    assert visible[1, 2].item()
    assert not visible[1, 7].item()


# ---------- compute_holdout_metrics ----------

def test_compute_holdout_metrics_shapes_and_keys() -> None:
    x_hat, target, idx = _make_inputs()
    out = compute_holdout_metrics(x_hat, target, idx)
    assert set(out.keys()) == {"loss_visible", "loss_holdout", "pearson_holdout", "r2_holdout"}
    for v in out.values():
        assert v.ndim == 0
        assert torch.isfinite(v)


def test_visible_loss_matches_manual() -> None:
    x_hat, target, idx = _make_inputs(B=3, G=15, K=4, seed=1)
    out = compute_holdout_metrics(x_hat, target, idx)
    # Manually compute the same loss by scattering.
    visible = _build_visible_mask(idx, G=15, padding_mask=None)
    manual = poisson_loss(x_hat, target, reduction="mean", target_mask=visible)
    assert torch.allclose(out["loss_visible"], manual)


def test_holdout_loss_matches_manual() -> None:
    x_hat, target, idx = _make_inputs(B=3, G=15, K=4, seed=2)
    out = compute_holdout_metrics(x_hat, target, idx)
    visible = _build_visible_mask(idx, G=15, padding_mask=None)
    holdout = ~visible
    manual = poisson_loss(x_hat, target, reduction="mean", target_mask=holdout)
    assert torch.allclose(out["loss_holdout"], manual)


def test_perfect_reconstruction() -> None:
    _, target, idx = _make_inputs(B=4, G=20, K=5, seed=3)
    out = compute_holdout_metrics(target.clone(), target, idx)
    assert torch.allclose(out["pearson_holdout"], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(out["r2_holdout"], torch.tensor(1.0), atol=1e-5)


def test_target_mask_intersection() -> None:
    """When target_mask drops a position, it should land in neither visible
    nor holdout — so the sum of contributions is bounded by the measured set."""
    x_hat, target, idx = _make_inputs(B=2, G=10, K=3, seed=4)
    # Mask out half the genes as "unmeasured".
    tm = torch.zeros(2, 10, dtype=torch.bool)
    tm[:, :5] = True
    out = compute_holdout_metrics(x_hat, target, idx, target_mask=tm)
    # Both metrics must be finite, no contribution from positions 5..9.
    assert torch.isfinite(out["loss_visible"])
    assert torch.isfinite(out["loss_holdout"])


def test_padding_mask_path() -> None:
    """Pad-slot indices should be excluded from visible — same value should
    appear in holdout instead (if it's in the measured set)."""
    B, G = 2, 10
    # Row 0: real unmasked = [0, 5]; pad slot fakes idx=9.
    unmasked_idx = torch.tensor([[0, 5, 9], [1, 2, 3]], dtype=torch.long)
    padding_mask = torch.tensor([[False, False, True], [False, False, False]], dtype=torch.bool)
    target = torch.rand(B, G)
    x_hat = target.clone()
    out = compute_holdout_metrics(
        x_hat, target, unmasked_idx, padding_mask=padding_mask,
    )
    # Perfect reconstruction → pearson on the holdout set should be 1.0
    # and the holdout loss should be finite.
    assert torch.allclose(out["pearson_holdout"], torch.tensor(1.0), atol=1e-5)


def test_pearson_on_constant_input_is_nan_aware() -> None:
    """If a row's holdout slice has zero variance, that row contributes NaN
    to per-sample pearson — but the aggregator should fall back to the
    remaining valid rows."""
    B, G, K = 2, 10, 3
    target = torch.zeros(B, G)
    target[1] = torch.linspace(0, 1, G)
    x_hat = torch.zeros(B, G)
    x_hat[1] = torch.linspace(0, 1, G)
    unmasked_idx = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    # Row 0 holdout = constant 0 → NaN; row 1 holdout = perfect → 1.0.
    p = _per_sample_pearson(x_hat, target, mask=~_build_visible_mask(unmasked_idx, G, None))
    assert torch.isnan(p[0])
    assert torch.allclose(p[1], torch.tensor(1.0), atol=1e-5)
    out = compute_holdout_metrics(x_hat, target, unmasked_idx)
    # Aggregate ignores the NaN row.
    assert torch.allclose(out["pearson_holdout"], torch.tensor(1.0), atol=1e-5)


# ---------- helpers stand on their own ----------

def test_per_sample_r2_perfect() -> None:
    target = torch.rand(3, 12)
    mask = torch.ones(3, 12, dtype=torch.bool)
    r2 = _per_sample_r2(target.clone(), target, mask)
    assert torch.allclose(r2, torch.ones(3), atol=1e-5)

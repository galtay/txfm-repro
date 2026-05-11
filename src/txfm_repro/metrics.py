"""Validation-time reconstruction metrics.

The training loss averages Poisson NLL over all G gene positions, which
conflates two qualitatively different things:

  - The K encoder-visible genes — the model literally sees their values as
    part of the token, so reconstructing them is a near-trivial passthrough.
  - The G - K masked-out genes — the actual reconstruction target.

This module computes both losses separately, plus per-sample Pearson and R²
on the held-out positions. Train side keeps logging scalar loss to stay
cheap; these run only inside `validation_step`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from txfm_repro.lit_model import poisson_loss


def _build_visible_mask(
    unmasked_idx: Tensor,
    G: int,
    padding_mask: Tensor | None,
) -> Tensor:
    """Scatter unmasked_idx into a (B, G) bool mask of encoder-visible positions.

    Pad slots (`padding_mask=True`) are excluded from the visible set so we
    don't accidentally mark gene 0 (the default fill for pad index) as visible.
    """
    B, K = unmasked_idx.shape
    visible = torch.zeros(B, G, dtype=torch.bool, device=unmasked_idx.device)
    if padding_mask is None:
        active = torch.ones_like(unmasked_idx, dtype=torch.bool)
    else:
        active = ~padding_mask
    # scatter active=True at unmasked_idx positions
    rows = torch.arange(B, device=unmasked_idx.device).unsqueeze(1).expand(B, K)
    visible[rows[active], unmasked_idx[active]] = True
    return visible


def _per_sample_pearson(
    x_hat: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Per-row Pearson on positions where `mask` is True. Rows with <2 valid
    positions return NaN; the caller averages over the rest."""
    B = x_hat.size(0)
    out = torch.full((B,), float("nan"), device=x_hat.device, dtype=x_hat.dtype)
    for b in range(B):
        m = mask[b]
        n = int(m.sum().item())
        if n < 2:
            continue
        a = x_hat[b][m]
        t = target[b][m]
        a = a - a.mean()
        t = t - t.mean()
        var_a = a.pow(2).sum()
        var_t = t.pow(2).sum()
        # Zero variance on either side → correlation is undefined; leave NaN.
        if var_a.item() < 1e-12 or var_t.item() < 1e-12:
            continue
        out[b] = (a * t).sum() / (var_a.sqrt() * var_t.sqrt())
    return out


def _per_sample_r2(
    x_hat: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Per-row coefficient of determination on positions where `mask` is True.

    R² = 1 - SS_res / SS_tot. Rows with <2 valid positions or zero variance
    return NaN."""
    B = x_hat.size(0)
    out = torch.full((B,), float("nan"), device=x_hat.device, dtype=x_hat.dtype)
    for b in range(B):
        m = mask[b]
        n = int(m.sum().item())
        if n < 2:
            continue
        a = x_hat[b][m]
        t = target[b][m]
        ss_res = (a - t).pow(2).sum()
        ss_tot = (t - t.mean()).pow(2).sum()
        if ss_tot.item() < 1e-12:
            continue
        out[b] = 1.0 - ss_res / ss_tot
    return out


def compute_holdout_metrics(
    x_hat: Tensor,
    target: Tensor,
    unmasked_idx: Tensor,
    padding_mask: Tensor | None = None,
    target_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return scalar tensors: loss_visible, loss_holdout, pearson_holdout, r2_holdout.

    Args:
      x_hat:        (B, G) reconstruction in log-rate space (output of model).
      target:       (B, G) library-normalized log1p target.
      unmasked_idx: (B, K) long, gene indices the encoder saw.
      padding_mask: (B, K) bool, True at pad slots in `unmasked_idx`.
      target_mask:  (B, G) bool, True at measured positions. When provided,
                    both visible and holdout masks are intersected with it
                    (unmeasured genes contribute to neither metric).

    The "visible" mask is the scattered set of unmasked_idx positions; the
    "holdout" mask is its complement within the measured set.
    """
    B, G = target.shape
    visible = _build_visible_mask(unmasked_idx, G, padding_mask)
    holdout = ~visible
    if target_mask is not None:
        visible = visible & target_mask
        holdout = holdout & target_mask

    loss_visible = poisson_loss(x_hat, target, reduction="mean", target_mask=visible)
    loss_holdout = poisson_loss(x_hat, target, reduction="mean", target_mask=holdout)

    pearson = _per_sample_pearson(x_hat, target, holdout)
    r2 = _per_sample_r2(x_hat, target, holdout)
    # nanmean — average over rows that had >=2 holdout positions.
    pearson_holdout = pearson[~torch.isnan(pearson)].mean() if (~torch.isnan(pearson)).any() else torch.tensor(float("nan"))
    r2_holdout = r2[~torch.isnan(r2)].mean() if (~torch.isnan(r2)).any() else torch.tensor(float("nan"))

    return {
        "loss_visible":     loss_visible,
        "loss_holdout":     loss_holdout,
        "pearson_holdout":  pearson_holdout,
        "r2_holdout":       r2_holdout,
    }

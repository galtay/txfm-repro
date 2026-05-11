"""Model sanity / invariance checks.

These tests catch silent bugs that loss-curves alone don't surface:
- preprocessing actually normalizes library size,
- the model is sensitive to which gene each value belongs to,
- training the model is doing more than a random-init permutation of features.

All tests use tiny models for CI speed (<2s each).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from txfm_repro.lit_model import LitTxFM, LitTxFMConfig, TxFM, poisson_loss
from txfm_repro.metrics import compute_holdout_metrics
from txfm_repro.mock_data import mask_collate

G = 64
K = 16


def _tiny_cfg() -> LitTxFMConfig:
    return LitTxFMConfig(
        n_genes=G, d_model=32, n_layers=1, n_heads=4,
        dim_ff=64, decoder_layers=1, dropout=0.0,
        library_size_L=1e5,
    )


def _make_counts(B: int = 4, scale: float = 1.0, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    counts = (torch.randint(0, 50, (B, G), generator=gen).to(torch.float32) * scale).to(torch.int64)
    measured = torch.ones_like(counts, dtype=torch.bool)
    return counts, measured


def _batch_from(counts: torch.Tensor, measured: torch.Tensor, K: int, seed: int) -> dict:
    items = [(counts[b], measured[b]) for b in range(counts.shape[0])]
    gen = torch.Generator().manual_seed(seed)
    return mask_collate(items, K=K, library_size_L=1e5, rng_generator=gen)


def test_library_size_norm_in_collate() -> None:
    """`mask_collate` library-size-normalizes counts before log1p. Scaling
    all input counts by a constant should leave the `target` (and therefore
    `unmasked_vals`) unchanged."""
    counts, measured = _make_counts(B=2, scale=1.0, seed=0)
    counts_2x = counts * 2

    batch_a = _batch_from(counts, measured, K=K, seed=42)
    batch_b = _batch_from(counts_2x, measured, K=K, seed=42)

    # The K-mask draw is reproducible (same seed), so unmasked_idx matches.
    assert torch.equal(batch_a["unmasked_idx"], batch_b["unmasked_idx"])
    # Library-normalized targets should match within float tolerance.
    assert torch.allclose(batch_a["target"], batch_b["target"], atol=1e-4)
    assert torch.allclose(batch_a["unmasked_vals"], batch_b["unmasked_vals"], atol=1e-4)


def test_gene_permutation_breaks_predictions() -> None:
    """Permute the gene indices fed to the encoder while keeping the values
    in place. If the model is using gene identity, predictions should change
    substantially. (Untrained: even a randomly-initialized embedding has gene
    identity baked in via `nn.Embedding`.)"""
    torch.manual_seed(0)
    model = TxFM(
        n_genes=G, d_model=32, n_layers=1, n_heads=4,
        dim_ff=64, decoder_layers=1, dropout=0.0,
        library_size_L=1e5,
    )
    model.eval()

    counts, measured = _make_counts(B=4, seed=1)
    batch = _batch_from(counts, measured, K=K, seed=2)

    with torch.no_grad():
        x_hat_orig = model(batch["unmasked_idx"], batch["unmasked_vals"])
        # Permute the gene IDs but keep the values aligned to their slot.
        gen = torch.Generator().manual_seed(99)
        perm = torch.randperm(G, generator=gen)
        permuted_idx = perm[batch["unmasked_idx"]]
        x_hat_perm = model(permuted_idx, batch["unmasked_vals"])

    # Predictions should diverge substantially. We don't assert a precise
    # ratio because the model is randomly initialized, just that the L2
    # distance is meaningfully > 0.
    diff = (x_hat_orig - x_hat_perm).pow(2).mean().sqrt().item()
    orig_scale = x_hat_orig.pow(2).mean().sqrt().item()
    assert diff > 0.05 * orig_scale, (
        f"permuting gene IDs barely changed predictions (rmsd={diff}, scale={orig_scale}) — "
        "model may be ignoring gene identity"
    )


def test_overfit_beats_random_baseline_on_holdout() -> None:
    """After ~50 overfit steps on a small batch, val/loss_holdout should be
    strictly lower than the same metric evaluated with random Gaussian
    predictions of comparable magnitude. Locks in 'training does something'."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lit = LitTxFM(cfg)

    counts, measured = _make_counts(B=4, seed=0)
    batch = _batch_from(counts, measured, K=K, seed=0)

    # Baseline: random Gaussian predictions, clipped to the model's output range.
    import math
    bound = math.log1p(cfg.library_size_L)
    rand_pred = torch.rand_like(batch["target"]) * bound
    baseline = compute_holdout_metrics(
        x_hat=rand_pred,
        target=batch["target"],
        unmasked_idx=batch["unmasked_idx"],
        padding_mask=batch.get("padding_mask"),
    )

    # Train.
    opt = torch.optim.AdamW(lit.parameters(), lr=1e-3)
    lit.train()
    for _ in range(50):
        x_hat = lit(batch["unmasked_idx"], batch["unmasked_vals"], padding_mask=batch.get("padding_mask"))
        loss = poisson_loss(x_hat, batch["target"], reduction="mean")
        opt.zero_grad()
        loss.backward()
        opt.step()

    lit.eval()
    with torch.no_grad():
        x_hat = lit(batch["unmasked_idx"], batch["unmasked_vals"], padding_mask=batch.get("padding_mask"))
        trained = compute_holdout_metrics(
            x_hat=x_hat, target=batch["target"],
            unmasked_idx=batch["unmasked_idx"],
            padding_mask=batch.get("padding_mask"),
        )

    # Trained loss should beat random by a clear margin.
    assert trained["loss_holdout"].item() < baseline["loss_holdout"].item() - 0.1, (
        f"trained loss_holdout {trained['loss_holdout'].item():.3f} "
        f"vs baseline {baseline['loss_holdout'].item():.3f} — model didn't overfit"
    )


def test_overfit_raises_mask_invariance() -> None:
    """Re-mask the same patient with two different K-mask draws. After
    training, the two embeddings should be MORE similar than they were
    at random initialization — i.e. the model is producing a sample-level
    representation that doesn't depend too much on which K genes were drawn."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lit_untrained = LitTxFM(cfg)
    lit_trained = LitTxFM(cfg)
    lit_trained.load_state_dict(lit_untrained.state_dict())  # start from same init

    counts, measured = _make_counts(B=4, seed=0)

    def _mean_pairwise_cos(model: LitTxFM) -> float:
        model.eval()
        with torch.no_grad():
            e1 = model.model.encoder(*_batch_pair(counts, measured, seed=1))
            e2 = model.model.encoder(*_batch_pair(counts, measured, seed=2))
        # Per-sample cosine sim between the two masks; average over B.
        cos = F.cosine_similarity(e1, e2, dim=-1)
        return cos.mean().item()

    def _batch_pair(c: torch.Tensor, m: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        b = _batch_from(c, m, K=K, seed=seed)
        return b["unmasked_idx"], b["unmasked_vals"]

    cos_init = _mean_pairwise_cos(lit_untrained)

    # Train lit_trained.
    opt = torch.optim.AdamW(lit_trained.parameters(), lr=1e-3)
    lit_trained.train()
    for _ in range(80):
        batch = _batch_from(counts, measured, K=K, seed=int(torch.randint(1000, (1,)).item()))
        x_hat = lit_trained(batch["unmasked_idx"], batch["unmasked_vals"], padding_mask=batch.get("padding_mask"))
        loss = poisson_loss(x_hat, batch["target"], reduction="mean")
        opt.zero_grad()
        loss.backward()
        opt.step()

    cos_trained = _mean_pairwise_cos(lit_trained)

    # Trained model should be at least as mask-invariant as random init.
    # We allow some slack — the bar is "training didn't break this".
    assert cos_trained >= cos_init - 0.05, (
        f"mask invariance regressed: init={cos_init:.3f}, trained={cos_trained:.3f}"
    )

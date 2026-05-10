"""Component shape/bound tests + an end-to-end overfit-a-batch smoke test.

The overfit test directly constructs LitTxFM and MockBulkDataModule rather
than going through the LightningCLI, so we only test architecture +
training-loop wiring here, not the CLI plumbing.
"""

from __future__ import annotations

import math

import lightning as L
import numpy as np
import pytest
import torch

from txfm_repro.lit_model import (
    LitTxFM,
    LitTxFMConfig,
    MLPDecoder,
    TxFM,
    TxFMEncoder,
    get_output_activation,
    poisson_loss,
    rectified_sigmoid,
    rectified_tanh,
)
from txfm_repro.mock_data import (
    MockBulkDataModule,
    MockBulkRNADataset,
    generate_mock_counts,
    generate_mock_counts_multi_source,
    mask_collate,
)


def test_rectified_tanh_bounds() -> None:
    L_val = 1e5
    z = torch.linspace(-100, 100, steps=257)
    out = rectified_tanh(z, L_val)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    assert (out <= math.log1p(L_val) + 1e-5).all()
    # Negative inputs clamp to exactly 0.
    assert torch.allclose(out[: out.numel() // 2 + 1].max(), torch.tensor(0.0))


def test_rectified_sigmoid_bounds() -> None:
    L_val = 1e5
    z = torch.linspace(-100, 100, steps=257)
    out = rectified_sigmoid(z, L_val)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    assert (out <= math.log1p(L_val) + 1e-5).all()
    assert torch.allclose(out[: out.numel() // 2 + 1].max(), torch.tensor(0.0))


def test_activation_equivalence() -> None:
    """tanh(z/(4e)) == 2*sigmoid(z/(2e)) - 1, so the two surface forms must
    agree across a wide range. Tighter tolerance than usual: this is a
    closed-form identity, not an approximation."""
    L_val = 1e5
    # Wide but float32-safe range; sigmoid/tanh saturate well before 100.
    z = torch.linspace(-50, 50, steps=4097, dtype=torch.float32)
    a = rectified_tanh(z, L_val)
    b = rectified_sigmoid(z, L_val)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (
        f"max abs diff = {(a - b).abs().max().item():.3e}"
    )


def test_activation_equivalence_float64() -> None:
    """Same equivalence at higher precision — confirms the identity holds
    independent of float32 rounding noise."""
    L_val = 1e5
    z = torch.linspace(-200, 200, steps=4097, dtype=torch.float64)
    a = rectified_tanh(z, L_val)
    b = rectified_sigmoid(z, L_val)
    assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)


def test_activation_gradient_match() -> None:
    """Gradients must agree too — they're the optimization signal."""
    L_val = 1e5
    z = torch.randn(64, requires_grad=True, dtype=torch.float64)
    rectified_tanh(z, L_val).sum().backward()
    g_tanh = z.grad.detach().clone()
    z.grad = None
    rectified_sigmoid(z, L_val).sum().backward()
    g_sigmoid = z.grad.detach().clone()
    assert torch.allclose(g_tanh, g_sigmoid, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_activation_extreme_values(activation: str) -> None:
    """No NaN/Inf at the saturation extremes the optimizer can drive `z` to."""
    L_val = 1e5
    fn = get_output_activation(activation)  # type: ignore[arg-type]
    z = torch.tensor([-1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6], dtype=torch.float32)
    out = fn(z, L_val)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    assert (out <= math.log1p(L_val) + 1e-5).all()
    # Saturation: large positive z should hit the upper bound.
    assert torch.isclose(out[-1], torch.tensor(math.log1p(L_val)), atol=1e-5)
    # Negative z must be exactly 0 (ReLU clamps the negative arm).
    assert torch.allclose(out[:3], torch.zeros(3))


def test_encoder_shape() -> None:
    enc = TxFMEncoder(n_genes=100, d_model=32, n_layers=1, n_heads=4, dim_ff=64, dropout=0.0)
    idx = torch.randint(0, 100, (2, 8))
    vals = torch.randn(2, 8)
    out = enc(idx, vals)
    assert out.shape == (2, 32)


def test_decoder_shape() -> None:
    dec = MLPDecoder(d_model=32, n_genes=100, decoder_layers=2, dropout=0.0)
    e = torch.randn(2, 32)
    out = dec(e)
    assert out.shape == (2, 100)


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_txfm_forward(activation: str) -> None:
    L_val = 1e5
    model = TxFM(
        n_genes=100,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dim_ff=64,
        decoder_layers=2,
        dropout=0.0,
        library_size_L=L_val,
        activation=activation,  # type: ignore[arg-type]
    )
    idx = torch.randint(0, 100, (2, 8))
    vals = torch.randn(2, 8)
    out = model(idx, vals)
    assert out.shape == (2, 100)
    assert (out >= 0).all()
    assert (out <= math.log1p(L_val) + 1e-5).all()


def test_litTxFM_activation_propagates() -> None:
    """LitTxFMConfig.activation must reach TxFM._activation."""
    cfg_t = LitTxFMConfig(n_genes=16, d_model=8, n_layers=1, n_heads=2, dim_ff=16,
                          decoder_layers=1, dropout=0.0, activation="tanh")
    cfg_s = LitTxFMConfig(n_genes=16, d_model=8, n_layers=1, n_heads=2, dim_ff=16,
                          decoder_layers=1, dropout=0.0, activation="sigmoid")
    lit_t = LitTxFM(cfg_t)
    lit_s = LitTxFM(cfg_s)
    assert lit_t.model.activation_name == "tanh"
    assert lit_s.model.activation_name == "sigmoid"
    assert lit_t.model._activation is rectified_tanh
    assert lit_s.model._activation is rectified_sigmoid


def test_poisson_loss_finite() -> None:
    x_hat = torch.randn(4, 64)
    target = torch.rand(4, 64) * math.log1p(1e5)
    loss = poisson_loss(x_hat, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_poisson_loss_reduction_consistency() -> None:
    """`reduction='mean'` and `'sum'` must agree with explicit reduction of
    the `'none'` form. Cheap but catches future op-rewrites that drift."""
    torch.manual_seed(0)
    x_hat = torch.randn(8, 32)
    target = torch.rand(8, 32) * math.log1p(1e5)

    none = poisson_loss(x_hat, target, reduction="none")
    mean = poisson_loss(x_hat, target, reduction="mean")
    summed = poisson_loss(x_hat, target, reduction="sum")

    assert none.shape == x_hat.shape
    assert torch.allclose(mean, none.mean(), atol=1e-6)
    assert torch.allclose(summed, none.sum(), atol=1e-4)


def test_poisson_loss_finite_at_realistic_extremes() -> None:
    """Under the rectified activation, x_hat is bounded in [0, log(L+1)] ≈
    [0, 11.5] for L=1e5. Targets live in the same log1p range. Confirm
    the loss stays finite across that joint domain plus a margin."""
    L_val = 1e5
    upper = math.log1p(L_val)
    n = 5000
    x_hat = torch.linspace(0, upper, n).repeat(2, 1)
    target = torch.linspace(0, upper, n).repeat(2, 1)
    loss = poisson_loss(x_hat, target, reduction="none")
    assert torch.isfinite(loss).all()


def test_poisson_loss_unknown_reduction() -> None:
    with pytest.raises(ValueError, match="unknown reduction"):
        poisson_loss(torch.zeros(2), torch.zeros(2), reduction="median")  # type: ignore[arg-type]


def test_get_output_activation_unknown() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        get_output_activation("relu")  # type: ignore[arg-type]


def _single_source_batch(n_samples: int, n_genes: int, seed: int = 0):
    counts = generate_mock_counts(n_samples=n_samples, n_genes=n_genes, seed=seed)
    ds = MockBulkRNADataset(counts)
    return [ds[i] for i in range(n_samples)]


def test_mask_collate_shapes() -> None:
    batch = _single_source_batch(n_samples=4, n_genes=64, seed=0)
    out = mask_collate(batch, K=8, library_size_L=1e5)
    assert out["unmasked_idx"].shape == (4, 8)
    assert out["unmasked_vals"].shape == (4, 8)
    assert out["target"].shape == (4, 64)
    assert out["padding_mask"].shape == (4, 8)
    assert out["target_mask"].shape == (4, 64)
    assert (out["target"] >= 0).all()
    # Single-source default: no padding, all genes measured.
    assert not out["padding_mask"].any()
    assert out["target_mask"].all()
    # Unmasked values must agree with the gathered targets.
    gathered = torch.gather(out["target"], 1, out["unmasked_idx"])
    assert torch.allclose(gathered, out["unmasked_vals"])


# ──────────────────────────────────────── padding mask ─────────────────────

def test_padding_mask_invariance() -> None:
    """Real tokens followed by padded slots must produce the same encoder
    output as those real tokens alone. If the padding mask isn't being
    respected, the random index/value content in pad slots will leak into
    attention and shift the CLS output."""
    torch.manual_seed(0)
    enc = TxFMEncoder(n_genes=100, d_model=32, n_layers=2, n_heads=4, dim_ff=64, dropout=0.0)
    enc.eval()
    idx_real = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    vals_real = torch.tensor([[0.5, 1.0, 1.5, 2.0]])

    # Same prefix, then 3 garbage pad slots.
    idx_padded = torch.cat([idx_real, torch.tensor([[77, 88, 99]])], dim=1)
    vals_padded = torch.cat([vals_real, torch.tensor([[100.0, -50.0, 7.0]])], dim=1)
    pad_mask = torch.tensor([[False, False, False, False, True, True, True]])

    with torch.no_grad():
        out_real = enc(idx_real, vals_real)
        out_padded = enc(idx_padded, vals_padded, padding_mask=pad_mask)

    assert torch.allclose(out_real, out_padded, atol=1e-5), (
        f"padded encoder output diverged from real-only by max abs "
        f"{(out_real - out_padded).abs().max().item():.3e}"
    )


def test_padding_mask_passes_through_full_model() -> None:
    """End-to-end: TxFM.forward + LitTxFM.forward both accept padding_mask."""
    cfg = LitTxFMConfig(n_genes=32, d_model=16, n_layers=1, n_heads=2, dim_ff=32,
                        decoder_layers=1, dropout=0.0)
    lit = LitTxFM(cfg)
    lit.eval()
    idx = torch.randint(0, 32, (3, 6))
    vals = torch.randn(3, 6)
    pad_mask = torch.tensor([
        [False, False, False, False, True,  True],
        [False, False, False, True,  True,  True],
        [False, False, False, False, False, True],
    ])
    with torch.no_grad():
        out = lit(idx, vals, padding_mask=pad_mask)
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()


# ──────────────────────────────────────── multi-source data ────────────────

def test_multi_source_distinct_measured_sets() -> None:
    """Each source draws its own random gene subset; samples assigned to
    different sources should have different measured sets."""
    counts, mask, source = generate_mock_counts_multi_source(
        n_samples=12, n_genes=80, n_sources=3, measured_fraction=0.6, seed=0
    )
    assert counts.shape == (12, 80)
    assert mask.shape == (12, 80)
    assert source.shape == (12,)
    # Round-robin → each source has 4 samples
    for s in range(3):
        assert (source == s).sum() == 4
    # Per-sample measured count = round(0.6 * 80) = 48
    n_meas = int(round(0.6 * 80))
    assert (mask.sum(axis=1) == n_meas).all()
    # Counts at unmeasured positions are zero
    assert (counts[~mask] == 0).all()
    # Different sources → different measured-set signatures
    sigs = {tuple(np.where(mask[i])[0]) for i in range(12)}
    assert len(sigs) == 3, f"expected 3 distinct measured sets across 3 sources, got {len(sigs)}"


def test_multi_source_backward_compat() -> None:
    """n_sources=1 with measured_fraction=1.0 → all-True measured mask."""
    _, mask, _ = generate_mock_counts_multi_source(
        n_samples=4, n_genes=20, n_sources=1, measured_fraction=1.0, seed=0
    )
    assert mask.all()


# ──────────────────────────────────────── variable K ────────────────────────

def test_variable_K_collate() -> None:
    """K_min < K → per-sample sequence length varies; padding_mask reflects it."""
    batch = _single_source_batch(n_samples=8, n_genes=50, seed=0)
    gen = torch.Generator().manual_seed(1)
    out = mask_collate(batch, K=20, library_size_L=1e5, K_min=5, rng_generator=gen)
    K_max = out["unmasked_idx"].shape[1]
    assert 5 <= K_max <= 20
    # At least one sample should have padded slots given the wide range.
    assert out["padding_mask"].any()
    # For each row, the padding_mask has a contiguous run of False then True.
    for b in range(8):
        m = out["padding_mask"][b]
        if m.any():
            first_pad = int(m.float().argmax())
            assert (m[:first_pad] == False).all()  # noqa: E712
            assert (m[first_pad:] == True).all()   # noqa: E712


def test_collate_caps_K_at_measured_count() -> None:
    """If a sample has fewer measured genes than K, K_i is clamped down."""
    n_samples, n_genes = 4, 100
    counts, mask, _ = generate_mock_counts_multi_source(
        n_samples=n_samples, n_genes=n_genes, n_sources=2, measured_fraction=0.1, seed=0,
    )
    ds = MockBulkRNADataset(counts, mask)
    batch = [ds[i] for i in range(n_samples)]
    # Each row has only 10 measured genes, but we ask for K=30.
    out = mask_collate(batch, K=30, library_size_L=1e5)
    K_max = out["unmasked_idx"].shape[1]
    assert K_max == 10, f"expected K to clamp to measured-count=10, got {K_max}"
    # No padding because every row uses all 10 measured slots.
    assert not out["padding_mask"].any()


# ──────────────────────────────────────── target mask in loss ──────────────

def test_target_mask_loss_uses_measured_only() -> None:
    """With target_mask, mean reduction normalizes by mask.sum(), not numel."""
    # All zero predictions and zero targets ⇒ per-element loss = e^0 - 0 = 1.
    x_hat = torch.zeros(3, 5)
    target = torch.zeros(3, 5)
    mask = torch.tensor([
        [True,  True,  False, False, False],
        [True,  False, False, False, False],
        [True,  True,  True,  False, False],
    ])
    # 6 measured positions, each contributing 1 → mean = 1.0
    loss = poisson_loss(x_hat, target, reduction="mean", target_mask=mask)
    assert torch.allclose(loss, torch.tensor(1.0))
    # Without the mask, mean over 15 positions all contributing 1 → also 1.0.
    loss_unmasked = poisson_loss(x_hat, target, reduction="mean")
    assert torch.allclose(loss_unmasked, torch.tensor(1.0))
    # But masked sum (6) differs from unmasked sum (15).
    s_masked = poisson_loss(x_hat, target, reduction="sum", target_mask=mask)
    s_unmasked = poisson_loss(x_hat, target, reduction="sum")
    assert torch.allclose(s_masked, torch.tensor(6.0))
    assert torch.allclose(s_unmasked, torch.tensor(15.0))


def test_target_mask_loss_zeros_unmeasured_per_elem() -> None:
    """With reduction='none' and a target_mask, masked positions are zeroed."""
    torch.manual_seed(0)
    x_hat = torch.randn(2, 6)
    target = torch.rand(2, 6) * 5.0
    mask = torch.tensor([[True, False, True, True, False, True],
                         [False, False, True, True, True, False]])
    loss = poisson_loss(x_hat, target, reduction="none", target_mask=mask)
    assert loss.shape == x_hat.shape
    assert (loss[~mask] == 0).all()
    assert (loss[mask] != 0).any()


# ──────────────────────────────────────── multi-source overfit ─────────────

def test_overfit_multi_source(tmp_path) -> None:
    """End-to-end with multiple sources + variable K. Padding mask + target
    mask paths must work together for training to converge on a small set."""
    L.seed_everything(0, workers=True)

    cfg = LitTxFMConfig(
        n_genes=80,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dim_ff=64,
        decoder_layers=1,
        dropout=0.0,
        library_size_L=1e5,
    )
    lit = LitTxFM(cfg)
    lit.configure_optimizers = lambda: torch.optim.AdamW(lit.parameters(), lr=3e-3)

    dm = MockBulkDataModule(
        n_genes=80,
        n_train=6,
        n_val=2,
        K_unmasked=12,
        K_unmasked_min=6,            # variable K per sample
        n_sources=3,
        measured_fraction=0.7,       # each source measures 56 of 80 genes
        library_size_L=1e5,
        batch_size=3,
        num_workers=0,
        seed=0,
    )

    losses: list[float] = []

    class _Capture(L.pytorch.callbacks.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            v = trainer.callback_metrics.get("train/loss_epoch")
            if v is not None:
                losses.append(float(v))

    trainer = L.Trainer(
        max_epochs=25,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        callbacks=[_Capture()],
    )
    trainer.fit(lit, datamodule=dm)

    assert len(losses) >= 2
    first, last = losses[0], losses[-1]
    assert math.isfinite(first) and math.isfinite(last)
    assert last < first * 0.6, (
        f"multi-source + variable-K overfit didn't converge: first={first} last={last}"
    )


@pytest.mark.parametrize("accelerator", ["cpu"])
def test_overfit_a_batch(accelerator: str, tmp_path) -> None:
    """Tiny end-to-end smoke: loss must drop substantially on a few samples
    repeated for many steps. Catches mask/loss-sign / activation regressions."""
    L.seed_everything(0, workers=True)

    cfg = LitTxFMConfig(
        n_genes=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dim_ff=64,
        decoder_layers=2,
        dropout=0.0,
        library_size_L=1e5,
    )
    lit = LitTxFM(cfg)
    # Lightning CLI normally configures the optimizer; here we configure it
    # explicitly on the model so the standalone Trainer.fit() path works.
    lit.configure_optimizers = lambda: torch.optim.AdamW(lit.parameters(), lr=3e-3)

    dm = MockBulkDataModule(
        n_genes=64,
        n_train=4,
        n_val=2,
        K_unmasked=8,
        library_size_L=1e5,
        batch_size=2,
        num_workers=0,
        seed=0,
    )

    losses: list[float] = []

    class _Capture(L.pytorch.callbacks.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            v = trainer.callback_metrics.get("train/loss_epoch")
            if v is not None:
                losses.append(float(v))

    trainer = L.Trainer(
        max_epochs=20,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        callbacks=[_Capture()],
    )
    trainer.fit(lit, datamodule=dm)

    assert len(losses) >= 2, "expected at least 2 epochs of train_loss_epoch logs"
    first, last = losses[0], losses[-1]
    assert math.isfinite(first) and math.isfinite(last)
    assert last < first * 0.5, f"loss should drop by >=50% on overfit; got first={first} last={last}"

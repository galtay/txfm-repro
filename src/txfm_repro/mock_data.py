"""Synthetic bulk RNA-seq counts + Lightning DataModule.

Counts are drawn from a Poisson with per-gene log-normal rates and per-sample
log-uniform library sizes. Real RNA-seq is over-dispersed and we'll need a
proper count generator later, but for Phase 0 a Poisson is enough structure
for the model to actually learn something — and our loss is Poisson-based
anyway, so the residual mismatch is small.

The data module also models a multi-source training corpus: each "source"
covers a different (random) subset of the global gene universe, and each
sample is assigned to a source. This lets us exercise the encoder's
padding-mask path and the loss's target-mask path on synthetic data
before we wire in real heterogeneous datasets.
"""

from __future__ import annotations

import functools

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def generate_mock_counts(
    n_samples: int,
    n_genes: int,
    seed: int,
    log_rate_mean: float = -7.0,
    log_rate_std: float = 2.0,
    log_libsize_low: float = 15.0,
    log_libsize_high: float = 18.5,
) -> np.ndarray:
    """Single-source mock counts: every sample measures every gene.

    Returned shape `(n_samples, n_genes)` int32. Used as the underlying
    rate-and-libsize generator for the multi-source variant; also kept as a
    standalone for the simple single-source path.
    """
    rng = np.random.default_rng(seed)
    log_rates = rng.normal(log_rate_mean, log_rate_std, size=n_genes)
    rates = np.exp(log_rates)
    rates = rates / rates.sum()  # global proportions

    log_libsizes = rng.uniform(log_libsize_low, log_libsize_high, size=n_samples)
    libsizes = np.exp(log_libsizes)

    expected = libsizes[:, None] * rates[None, :]
    counts = rng.poisson(expected).astype(np.int32)
    return counts


def generate_mock_counts_multi_source(
    n_samples: int,
    n_genes: int,
    n_sources: int,
    measured_fraction: float,
    seed: int,
    log_rate_mean: float = -7.0,
    log_rate_std: float = 2.0,
    log_libsize_low: float = 15.0,
    log_libsize_high: float = 18.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-source mock counts.

    Each of `n_sources` "data sources" picks a random subset of size
    `round(measured_fraction * n_genes)` from the global gene universe; that
    subset is the source's measured gene set. Every sample is assigned to a
    source (round-robin) and only carries non-zero counts at its source's
    measured positions. The remaining positions are masked off in the
    returned `measured_mask` so downstream code (loss, library-size norm)
    can ignore them.

    Returns:
      counts:         (n_samples, n_genes) int32, zero at unmeasured positions
      measured_mask:  (n_samples, n_genes) bool, True at measured positions
      sample_source:  (n_samples,) int32, which source each sample came from

    Backward-compat note: `n_sources=1, measured_fraction=1.0` produces a
    measured_mask of all True and a counts matrix that matches what
    `generate_mock_counts` would produce for the same seed (modulo rng draws).
    """
    if n_sources < 1:
        raise ValueError(f"n_sources must be >= 1, got {n_sources}")
    if not (0.0 < measured_fraction <= 1.0):
        raise ValueError(f"measured_fraction must be in (0, 1], got {measured_fraction}")

    rng = np.random.default_rng(seed)
    n_measured = max(1, int(round(measured_fraction * n_genes)))

    # Per-source measured-gene index sets (sorted, distinct draws).
    source_measured: list[np.ndarray] = []
    for _ in range(n_sources):
        idx = rng.choice(n_genes, size=n_measured, replace=False)
        idx.sort()
        source_measured.append(idx)

    # Round-robin sample → source assignment so train/val cover every source
    # even at small N. (Random assignment is also fine but RR is deterministic
    # given seed and gives even per-source counts.)
    sample_source = (np.arange(n_samples) % n_sources).astype(np.int32)

    # Global gene rates (defined over the full universe). For each source we
    # renormalize over its measured subset so the source's library size is
    # proportional to those genes only.
    log_rates = rng.normal(log_rate_mean, log_rate_std, size=n_genes)
    rates = np.exp(log_rates)
    rates = rates / rates.sum()

    log_libsizes = rng.uniform(log_libsize_low, log_libsize_high, size=n_samples)
    libsizes = np.exp(log_libsizes)

    counts = np.zeros((n_samples, n_genes), dtype=np.int32)
    measured_mask = np.zeros((n_samples, n_genes), dtype=bool)

    for i in range(n_samples):
        s = sample_source[i]
        idx = source_measured[s]
        src_rates = rates[idx]
        src_rates = src_rates / src_rates.sum()
        counts[i, idx] = rng.poisson(libsizes[i] * src_rates).astype(np.int32)
        measured_mask[i, idx] = True

    return counts, measured_mask, sample_source


class MockBulkRNADataset(Dataset):
    """Holds a `(counts, measured_mask)` pair per sample.

    `measured_mask=None` is shorthand for "every gene measured" (single-source
    backward-compat path).
    """

    def __init__(
        self,
        counts: np.ndarray,
        measured_mask: np.ndarray | None = None,
    ) -> None:
        # int64 for embedding-lookup ergonomics on the gather path.
        self.counts = torch.from_numpy(counts).to(torch.int64)
        if measured_mask is None:
            measured_mask = np.ones_like(counts, dtype=bool)
        self.measured_mask = torch.from_numpy(measured_mask).to(torch.bool)

    def __len__(self) -> int:
        return self.counts.shape[0]

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.counts[i], self.measured_mask[i]


def mask_collate(
    batch: list[tuple[Tensor, Tensor]],
    K: int,
    library_size_L: float,
    K_min: int | None = None,
    rng_generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Library-size norm + log1p + per-sample K-of-measured gather, with padding.

    Each batch item is `(counts, measured_mask)`. Library size is summed over
    measured positions only. Each row picks `K_i` indices from its measured
    set; `K_i` is `K` by default, or sampled uniformly from `[K_min, K]` when
    `K_min` is given (variable sequence length). `K_i` is clamped to the
    number of measured positions on that row, so a sparse-coverage source
    can't ask for more tokens than it has.

    Returns a dict with both `padding_mask` (encoder-side) and `target_mask`
    (loss-side). When the batch is uniform (all rows full coverage and same K)
    these masks are still emitted: `padding_mask` is all-False, `target_mask`
    is all-True. Downstream code is shape-stable either way.
    """
    counts = torch.stack([b[0] for b in batch], dim=0)         # (B, G) int64
    measured = torch.stack([b[1] for b in batch], dim=0)        # (B, G) bool
    B, G = counts.shape
    if K > G:
        raise ValueError(f"K={K} cannot exceed n_genes={G}")
    if K_min is not None and not (0 < K_min <= K):
        raise ValueError(f"K_min must satisfy 0 < K_min <= K, got K_min={K_min}, K={K}")

    counts_f = counts.to(torch.float32)
    measured_f = measured.to(torch.float32)
    # Library size over MEASURED positions only — unmeasured zeros would
    # otherwise still be summed but we prefer to be explicit.
    lib = (counts_f * measured_f).sum(dim=-1, keepdim=True).clamp_min(1)
    target = torch.log1p(counts_f * library_size_L / lib)
    target = target * measured_f  # zero at unmeasured positions

    measured_counts = measured.sum(dim=-1).long()  # (B,)
    if K_min is not None:
        K_per_sample = torch.randint(K_min, K + 1, (B,), generator=rng_generator)
    else:
        K_per_sample = torch.full((B,), K, dtype=torch.long)
    K_per_sample = torch.minimum(K_per_sample, measured_counts)
    K_max = int(K_per_sample.max().item())
    if K_max == 0:
        raise ValueError("no measured genes in this batch — every sample has empty coverage")

    unmasked_idx = torch.zeros(B, K_max, dtype=torch.long)
    unmasked_vals = torch.zeros(B, K_max, dtype=torch.float32)
    padding_mask = torch.ones(B, K_max, dtype=torch.bool)  # True = padded slot

    for b in range(B):
        measured_indices = measured[b].nonzero(as_tuple=True)[0]
        Ki = int(K_per_sample[b].item())
        if Ki == 0:
            continue
        perm = torch.randperm(len(measured_indices), generator=rng_generator)[:Ki]
        chosen = measured_indices[perm]
        unmasked_idx[b, :Ki] = chosen
        unmasked_vals[b, :Ki] = target[b, chosen]
        padding_mask[b, :Ki] = False

    return {
        "unmasked_idx":   unmasked_idx,    # (B, K_max) long
        "unmasked_vals":  unmasked_vals,   # (B, K_max) float32
        "padding_mask":   padding_mask,    # (B, K_max) bool, True at pad
        "target":         target,          # (B, G) float32
        "target_mask":    measured,        # (B, G) bool, True at measured
        "library_size":   lib.squeeze(-1), # (B,)
    }


class MockBulkDataModule(L.LightningDataModule):
    """Flat init signature so Lightning CLI can expose every knob as a `data.*`
    argument and YAML key without needing a wrapper config object.

    Defaults: `n_sources=1, measured_fraction=1.0, K_unmasked_min=None` —
    i.e. one source measuring every gene with fixed-K masking, which matches
    the original single-source behavior.
    """

    def __init__(
        self,
        n_genes: int = 2000,
        n_train: int = 512,
        n_val: int = 64,
        K_unmasked: int = 256,
        K_unmasked_min: int | None = None,
        n_sources: int = 1,
        measured_fraction: float = 1.0,
        library_size_L: float = 1e5,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_genes = n_genes
        self.n_train = n_train
        self.n_val = n_val
        self.K_unmasked = K_unmasked
        self.K_unmasked_min = K_unmasked_min
        self.n_sources = n_sources
        self.measured_fraction = measured_fraction
        self.library_size_L = library_size_L
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self._train: MockBulkRNADataset | None = None
        self._val: MockBulkRNADataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train is None:
            counts, mask, _ = generate_mock_counts_multi_source(
                n_samples=self.n_train,
                n_genes=self.n_genes,
                n_sources=self.n_sources,
                measured_fraction=self.measured_fraction,
                seed=self.seed,
            )
            self._train = MockBulkRNADataset(counts, mask)
        if self._val is None:
            counts, mask, _ = generate_mock_counts_multi_source(
                n_samples=self.n_val,
                n_genes=self.n_genes,
                n_sources=self.n_sources,
                measured_fraction=self.measured_fraction,
                seed=self.seed + 1,
            )
            self._val = MockBulkRNADataset(counts, mask)

    def _collate(self):
        return functools.partial(
            mask_collate,
            K=self.K_unmasked,
            library_size_L=self.library_size_L,
            K_min=self.K_unmasked_min,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
        )

"""Offline tests for the TCGA streaming DataModule.

Network is gated behind `@pytest.mark.network` and the `TCGA_NETWORK_TESTS=1`
env var so CI/local runs don't hit HuggingFace by default. The bulk of the
logic (hash split, contract shape, empty-row skip, collate compat) is
exercised against an in-memory fake row generator via the dataset's test hook.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
import torch

from txfm_repro.lit_model import LitTxFM, LitTxFMConfig
from txfm_repro.mock_data import mask_collate
from txfm_repro.tcga_data import (
    TCGADataModule,
    TCGAStreamingIterableDataset,
    _patient_hash_bucket,
    freeze_gene_universe,
)

G = 64  # tiny gene universe for the fake stream — keeps fixtures fast


def _fake_row(case_id: str, n_aliquots: int, counts_seed: int) -> dict:
    rng = torch.Generator().manual_seed(counts_seed)
    aliquots = []
    for _ in range(n_aliquots):
        # Use a Poisson-ish int distribution so library size is realistic.
        counts = torch.randint(0, 50, (G,), generator=rng).tolist()
        aliquots.append({
            "sample_id": "s",
            "aliquot_id": "a",
            "source_file_id": "f",
            "gene_id": [f"ENSG{i:011d}.1" for i in range(G)],
            "gene_name": [f"GENE{i}" for i in range(G)],
            "gene_type": ["protein_coding"] * G,
            "unstranded": counts,
        })
    return {
        "case_id": case_id,
        "project_id": "TCGA-LUAD",
        "samples_gene_expression_quantification": aliquots,
    }


def _fake_rows(n_patients: int, n_buckets: int) -> list[dict]:
    rows: list[dict] = []
    # Generate UUIDs deterministically so the bucket distribution is the same
    # across pytest runs even though uuid.uuid4 is random.
    rng = torch.Generator().manual_seed(0)
    for i in range(n_patients):
        # 16 random bytes → string repr
        ints = torch.randint(0, 256, (16,), generator=rng, dtype=torch.uint8).tolist()
        cid = str(uuid.UUID(bytes=bytes(ints)))
        rows.append(_fake_row(cid, n_aliquots=1, counts_seed=i))
    return rows


# ---------- _patient_hash_bucket ----------

def test_patient_hash_bucket_deterministic() -> None:
    cid = "07b5663f-9a54-4462-b6c1-6fc8116b8714"
    assert _patient_hash_bucket(cid, 10) == _patient_hash_bucket(cid, 10)


def test_patient_hash_split_disjoint() -> None:
    case_ids = [str(uuid.uuid4()) for _ in range(100)]
    train, val = set(), set()
    for cid in case_ids:
        if _patient_hash_bucket(cid, 10) == 0:
            val.add(cid)
        else:
            train.add(cid)
    assert train.isdisjoint(val)
    assert train | val == set(case_ids)


def test_patient_hash_split_fraction() -> None:
    case_ids = [str(uuid.uuid4()) for _ in range(2000)]
    n_val = sum(1 for c in case_ids if _patient_hash_bucket(c, 10) == 0)
    frac = n_val / len(case_ids)
    # ~10% expected; allow a loose tolerance because 2000 is small.
    assert 0.07 < frac < 0.13, f"unexpected val fraction {frac}"


# ---------- freeze_gene_universe ----------

def test_freeze_gene_universe_loads_cache(tmp_path) -> None:
    cache = tmp_path / "gene_ids.json"
    gene_ids = [f"ENSG{i:011d}.1" for i in range(32)]
    cache.write_text(json.dumps(gene_ids))
    # Should NOT hit HF — pass a nonsense repo to assert no network call.
    out = freeze_gene_universe("not/a/real/repo", "TCGA-NONE", cache_path=cache)
    assert out == gene_ids


# ---------- TCGAStreamingIterableDataset ----------

def test_iterable_dataset_emits_contract() -> None:
    rows = _fake_rows(20, n_buckets=10)
    ds = TCGAStreamingIterableDataset(
        hf_repo="x", hf_config="x",
        gene_id_list=[f"ENSG{i:011d}.1" for i in range(G)],
        split="train", val_hash_bucket=0, n_hash_buckets=10,
        _source_iterable_factory=lambda: iter(rows),
    )
    items = list(ds)
    assert len(items) > 0
    for counts, mask in items:
        assert counts.dtype == torch.int64
        assert mask.dtype == torch.bool
        assert counts.shape == (G,)
        assert mask.shape == (G,)
        assert mask.all()
        assert counts.sum().item() > 0


def test_iterable_dataset_skips_empty_aliquots() -> None:
    rows = [
        _fake_row("case-a", n_aliquots=1, counts_seed=1),
        {  # empty aliquot list — should be skipped
            "case_id": "case-b",
            "project_id": "TCGA-LUAD",
            "samples_gene_expression_quantification": [],
        },
        _fake_row("case-c", n_aliquots=1, counts_seed=3),
    ]
    ds = TCGAStreamingIterableDataset(
        hf_repo="x", hf_config="x",
        gene_id_list=[f"ENSG{i:011d}.1" for i in range(G)],
        split="train", val_hash_bucket=999,  # so no row is filtered into val
        n_hash_buckets=1000,
        _source_iterable_factory=lambda: iter(rows),
    )
    items = list(ds)
    # Two of three rows have aliquots; both fall in "train" (val bucket is 999 of 1000).
    assert len(items) == 2


def test_iterable_dataset_split_filter() -> None:
    rows = _fake_rows(200, n_buckets=10)
    train_ds = TCGAStreamingIterableDataset(
        hf_repo="x", hf_config="x",
        gene_id_list=[f"ENSG{i:011d}.1" for i in range(G)],
        split="train", val_hash_bucket=0, n_hash_buckets=10,
        _source_iterable_factory=lambda: iter(rows),
    )
    val_ds = TCGAStreamingIterableDataset(
        hf_repo="x", hf_config="x",
        gene_id_list=[f"ENSG{i:011d}.1" for i in range(G)],
        split="val", val_hash_bucket=0, n_hash_buckets=10,
        _source_iterable_factory=lambda: iter(rows),
    )
    n_train = sum(1 for _ in train_ds)
    n_val = sum(1 for _ in val_ds)
    assert n_train + n_val == len(rows)
    assert n_val > 0
    assert 0.05 < n_val / len(rows) < 0.20


def test_iterable_dataset_max_rows() -> None:
    rows = _fake_rows(50, n_buckets=10)
    ds = TCGAStreamingIterableDataset(
        hf_repo="x", hf_config="x",
        gene_id_list=[f"ENSG{i:011d}.1" for i in range(G)],
        split="train", val_hash_bucket=0, n_hash_buckets=10,
        max_rows=5,
        _source_iterable_factory=lambda: iter(rows),
    )
    assert len(list(ds)) == 5


def test_aliquot_strategy_random_raises() -> None:
    with pytest.raises(NotImplementedError):
        TCGAStreamingIterableDataset(
            hf_repo="x", hf_config="x",
            gene_id_list=[f"ENSG{i:011d}.1" for i in range(G)],
            split="train",
            aliquot_strategy="random",
            _source_iterable_factory=lambda: iter([]),
        )


# ---------- collate compatibility with the model ----------

def test_tcga_collate_compat_with_lit_model() -> None:
    """A small end-to-end smoke: fake rows → IterableDataset → mask_collate →
    LitTxFM.training_step(). Catches contract drift between the data module
    and the model in one shot."""
    rows = _fake_rows(8, n_buckets=10)
    ds = TCGAStreamingIterableDataset(
        hf_repo="x", hf_config="x",
        gene_id_list=[f"ENSG{i:011d}.1" for i in range(G)],
        split="train", val_hash_bucket=999, n_hash_buckets=1000,
        _source_iterable_factory=lambda: iter(rows),
    )
    items = list(ds)
    assert len(items) >= 4
    batch = mask_collate(items[:4], K=8, library_size_L=1e5)
    assert batch["unmasked_idx"].shape == (4, 8)
    assert batch["unmasked_vals"].shape == (4, 8)
    assert batch["target"].shape == (4, G)
    assert batch["target_mask"].shape == (4, G)
    assert batch["target_mask"].all()
    assert torch.isfinite(batch["target"]).all()

    cfg = LitTxFMConfig(n_genes=G, d_model=16, n_layers=1, n_heads=2,
                        dim_ff=32, decoder_layers=1, dropout=0.0)
    lit = LitTxFM(cfg)
    lit.train()
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss)


# ---------- TCGADataModule wiring ----------

def test_datamodule_setup_asserts_gene_count(tmp_path) -> None:
    cache = tmp_path / "gene_ids.json"
    cache.write_text(json.dumps([f"ENSG{i:011d}.1" for i in range(G)]))
    dm = TCGADataModule(
        n_genes=G,
        gene_id_cache_path=str(cache),
        K_unmasked=4,
        batch_size=2,
    )
    dm.setup()
    assert dm.gene_id_list is not None
    assert len(dm.gene_id_list) == G

    bad = TCGADataModule(
        n_genes=G + 1,  # intentional mismatch
        gene_id_cache_path=str(cache),
        K_unmasked=4,
        batch_size=2,
    )
    with pytest.raises(ValueError):
        bad.setup()


# ---------- opt-in real-network smoke ----------

@pytest.mark.network
@pytest.mark.skipif(
    os.getenv("TCGA_NETWORK_TESTS") != "1",
    reason="set TCGA_NETWORK_TESTS=1 to hit HuggingFace",
)
def test_real_stream_smoke() -> None:
    from datasets import load_dataset
    ds = load_dataset(
        "gabrielaltay/tcga-patients-open", "TCGA-LUAD",
        split="train", streaming=True,
    )
    rows_seen = 0
    for row in ds:
        aliquots = row.get("samples_gene_expression_quantification") or []
        if not aliquots:
            continue
        a = aliquots[0]
        assert len(a["gene_id"]) == 60660
        assert sum(a["unstranded"]) > 1_000_000
        rows_seen += 1
        if rows_seen >= 2:
            break
    assert rows_seen == 2

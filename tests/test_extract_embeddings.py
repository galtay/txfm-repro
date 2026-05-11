"""Offline test for scripts/extract_embeddings.py — uses the
_source_iterable_factory hook to fake an HF stream and saves a checkpoint
inline so no network or training run is needed."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_embeddings import extract  # noqa: E402

from txfm_repro.lit_model import LitTxFM, LitTxFMConfig

G = 64


def _fake_row(case_id: str, sample_type: str, os_event: int, os_time: float) -> dict:
    rng = torch.Generator().manual_seed(abs(hash(case_id)) % (2**31))
    counts = torch.randint(0, 50, (G,), generator=rng).tolist()
    return {
        "case_id":      case_id,
        "project_id":   "TCGA-LUAD",
        "primary_site": "Lung",
        "disease_type": "Adenomas and Adenocarcinomas",
        "samples": [{"sample_type": sample_type}],
        "survival_derived": {"os_event": os_event, "os_time": os_time},
        "samples_gene_expression_quantification": [{
            "gene_id":    [f"ENSG{i:011d}.1" for i in range(G)],
            "gene_name":  [f"GENE{i}" for i in range(G)],
            "unstranded": counts,
        }],
    }


def _save_tiny_checkpoint(tmp_path: Path) -> Path:
    """Build a tiny LitTxFM and save it via Lightning's save_checkpoint API,
    so that LitTxFM.load_from_checkpoint can read it back."""
    cfg = LitTxFMConfig(
        n_genes=G, d_model=16, n_layers=1, n_heads=2,
        dim_ff=32, decoder_layers=1, dropout=0.0,
    )
    lit = LitTxFM(cfg)

    import lightning as L
    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )
    # Build a no-op dataset that produces one batch so Trainer can attach.
    from torch.utils.data import DataLoader, TensorDataset
    dummy = TensorDataset(torch.zeros(1, 1))

    class _Loop(L.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(dummy, batch_size=1, collate_fn=lambda b: None)

    # Skip Trainer.fit (we don't need actual training, just a saveable model).
    # Trainer.save_checkpoint can be called without fit.
    trainer.strategy.connect(lit)
    ckpt_path = tmp_path / "tiny.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path


def test_extract_writes_parquet_with_expected_columns(tmp_path) -> None:
    # Tiny in-memory dataset of 6 patients, 3 tumor + 3 normal, no real HF.
    rows = [
        _fake_row(str(uuid.UUID(int=i)), st, ev, tm)
        for i, (st, ev, tm) in enumerate([
            ("Primary Tumor", 1, 500.0),
            ("Primary Tumor", 0, 1200.0),
            ("Primary Tumor", 1, 300.0),
            ("Solid Tissue Normal", 0, 1500.0),
            ("Solid Tissue Normal", 0, 1800.0),
            ("Primary Tumor", 1, 700.0),
        ])
    ]

    # Frozen gene_id cache.
    cache = tmp_path / "gene_ids.json"
    cache.write_text(json.dumps([f"ENSG{i:011d}.1" for i in range(G)]))

    ckpt = _save_tiny_checkpoint(tmp_path)
    out = tmp_path / "embeddings.parquet"

    extract(
        ckpt_path=ckpt,
        hf_repo="x", hf_config="x",
        out_path=out,
        K=8, K_min=None, library_size_L=1e5, batch_size=2,
        n_hash_buckets=10,
        gene_id_cache_path=cache,
        seed=0, max_rows=None, device="cpu",
        _source_iterable_factory=lambda: iter(rows),
    )

    df = pd.read_parquet(out)
    assert len(df) == 6
    expected_meta = {"case_id", "project_id", "primary_site", "disease_type",
                     "sample_type", "os_event", "os_time", "hash_bucket"}
    assert expected_meta.issubset(set(df.columns))
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    assert len(emb_cols) == 16  # d_model
    # Embedding values are floats, not all-zero (tiny model is still nontrivial).
    emb = df[emb_cols].to_numpy()
    assert emb.shape == (6, 16)
    assert (emb.std(axis=0) > 0).any()


def test_extract_skips_empty_aliquot_rows(tmp_path) -> None:
    rows = [
        _fake_row("case-a", "Primary Tumor", 1, 100.0),
        {  # empty aliquots
            "case_id": "case-b",
            "samples_gene_expression_quantification": [],
            "samples": [{"sample_type": "x"}],
            "survival_derived": {"os_event": 0, "os_time": 0.0},
        },
        _fake_row("case-c", "Primary Tumor", 0, 200.0),
    ]
    cache = tmp_path / "gene_ids.json"
    cache.write_text(json.dumps([f"ENSG{i:011d}.1" for i in range(G)]))
    ckpt = _save_tiny_checkpoint(tmp_path)
    out = tmp_path / "embeddings.parquet"

    extract(
        ckpt_path=ckpt,
        hf_repo="x", hf_config="x",
        out_path=out,
        K=8, K_min=None, library_size_L=1e5, batch_size=2,
        n_hash_buckets=10,
        gene_id_cache_path=cache,
        seed=0, max_rows=None, device="cpu",
        _source_iterable_factory=lambda: iter(rows),
    )
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert set(df["case_id"]) == {"case-a", "case-c"}

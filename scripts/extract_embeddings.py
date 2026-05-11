"""Extract CLS embeddings for every patient in a TCGA project.

Loads a trained `LitTxFM` checkpoint, streams the full HF dataset for one
project, runs the encoder on each patient's first aliquot (with a fixed
RNG seed for deterministic K-masking), joins each `(case_id, embedding)`
with the row's clinical metadata, and dumps a parquet.

Output columns:
  case_id, project_id, primary_site, disease_type, sample_type,
  os_event, os_time, hash_bucket, embedding_0 ... embedding_{d-1}

The `hash_bucket` column matches the training-time train/val partition
(same blake2b hash, same `n_hash_buckets`) so probe scripts can split
embeddings the same way the model did.

Run:
  uv run python scripts/extract_embeddings.py \\
      --ckpt lightning_logs/version_N/checkpoints/last.ckpt \\
      --hf-config TCGA-LUAD \\
      --out out/embeddings_luad.parquet
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Allow `uv run python scripts/extract_embeddings.py` without `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from txfm_repro.lit_model import LitTxFM  # noqa: E402
from txfm_repro.mock_data import metadata_collate  # noqa: E402
from txfm_repro.tcga_data import (  # noqa: E402
    TCGAStreamingIterableDataset,
    _patient_hash_bucket,
    freeze_gene_universe,
)


def _first_sample_type(row: dict) -> str | None:
    """Pull `sample_type` from the first sample record on the row. The TCGA
    schema puts this at row.samples[i].sample_type — typically values like
    'Primary Tumor' or 'Solid Tissue Normal'."""
    samples = row.get("samples") or []
    if not samples:
        return None
    return samples[0].get("sample_type")


def _row_metadata(row: dict, hash_bucket: int) -> dict:
    surv = row.get("survival_derived") or {}
    return {
        "case_id":       row.get("case_id"),
        "project_id":    row.get("project_id"),
        "primary_site":  row.get("primary_site"),
        "disease_type":  row.get("disease_type"),
        "sample_type":   _first_sample_type(row),
        "os_event":      surv.get("os_event"),
        "os_time":       surv.get("os_time"),
        "hash_bucket":   hash_bucket,
    }


def extract(
    ckpt_path: Path,
    hf_repo: str,
    hf_config: str,
    out_path: Path,
    K: int,
    K_min: int | None,
    library_size_L: float,
    batch_size: int,
    n_hash_buckets: int,
    gene_id_cache_path: Path,
    seed: int,
    max_rows: int | None,
    device: str,
    _source_iterable_factory=None,  # test hook
    _row_factory=None,               # test hook for metadata rows (mirrors counts source)
) -> Path:
    gene_id_list = freeze_gene_universe(hf_repo, hf_config, cache_path=gene_id_cache_path)
    G = len(gene_id_list)

    lit = LitTxFM.load_from_checkpoint(ckpt_path, map_location=device)
    lit.eval()
    lit.freeze()

    # We need two iterators in lockstep: one yielding (counts, mask, case_id)
    # for the model, one yielding the full row dict for metadata. Easiest is
    # to iterate the HF source ourselves, build tuples, AND keep a parallel
    # dict of case_id -> metadata.
    metadata_by_case: dict[str, dict] = {}

    if _source_iterable_factory is not None:
        source_factory = _source_iterable_factory
    else:
        from datasets import load_dataset
        def source_factory():
            return load_dataset(hf_repo, hf_config, split="train", streaming=True)

    def annotated_source():
        for row in source_factory():
            cid = row.get("case_id")
            if cid is None:
                continue
            aliquots = row.get("samples_gene_expression_quantification") or []
            if not aliquots:
                continue
            bucket = _patient_hash_bucket(cid, n_hash_buckets)
            metadata_by_case[cid] = _row_metadata(row, bucket)
            yield row

    ds = TCGAStreamingIterableDataset(
        hf_repo=hf_repo,
        hf_config=hf_config,
        gene_id_list=gene_id_list,
        split="all",
        n_hash_buckets=n_hash_buckets,
        max_rows=max_rows,
        with_case_id=True,
        _source_iterable_factory=annotated_source,
    )
    rng = torch.Generator().manual_seed(seed)
    collate = functools.partial(
        metadata_collate, K=K, library_size_L=library_size_L, K_min=K_min, rng_generator=rng,
    )
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=collate)

    rows: list[dict] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_dev = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            out = lit.predict_step(batch_dev, batch_idx)
            emb = out["embedding"].cpu().to(torch.float32)
            case_ids: list[str] = out["case_id"]
            for i, cid in enumerate(case_ids):
                meta = metadata_by_case.get(cid, {"case_id": cid})
                row_out = dict(meta)
                row_out.update({f"embedding_{j}": float(emb[i, j]) for j in range(emb.shape[1])})
                rows.append(row_out)

    if not rows:
        raise RuntimeError("no embeddings produced — empty stream or no aliquots")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--hf-repo", default="gabrielaltay/tcga-patients-open")
    p.add_argument("--hf-config", default="TCGA-LUAD")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--K-unmasked", type=int, default=1024)
    p.add_argument("--K-unmasked-min", type=int, default=None,
                   help="omit (or null) for fixed K; useful when checkpoint trained on variable-K")
    p.add_argument("--library-size-L", type=float, default=1e5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-hash-buckets", type=int, default=10)
    p.add_argument("--gene-id-cache-path", type=Path, default=Path("configs/gene_ids_gencode_v36.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    args = p.parse_args()

    out = extract(
        ckpt_path=args.ckpt,
        hf_repo=args.hf_repo,
        hf_config=args.hf_config,
        out_path=args.out,
        K=args.K_unmasked,
        K_min=args.K_unmasked_min,
        library_size_L=args.library_size_L,
        batch_size=args.batch_size,
        n_hash_buckets=args.n_hash_buckets,
        gene_id_cache_path=args.gene_id_cache_path,
        seed=args.seed,
        max_rows=args.max_rows,
        device=args.device,
    )
    print(f"wrote {out}")
    df = pd.read_parquet(out)
    print(f"rows: {len(df)}")
    print(f"non-embedding columns: {[c for c in df.columns if not c.startswith('embedding_')]}")
    if "sample_type" in df:
        print("sample_type counts:")
        print(df["sample_type"].value_counts(dropna=False).to_string())
    if "hash_bucket" in df:
        print("hash_bucket counts:")
        print(df["hash_bucket"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

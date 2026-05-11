"""Phase 1 sandbox: stitch IterableDataset → DataLoader → mask_collate → LitTxFM.

Pulls one real batch through the existing `mask_collate` pipeline, prints
every key in the returned dict, then drives a single `LitTxFM.training_step`
on the batch to prove the data contract matches the model.

Run: `uv run python dev_sandbox/04_collate_through_dataloader.py`
"""

from __future__ import annotations

import functools
import itertools
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

# Make the package importable when running directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from txfm_repro.lit_model import LitTxFM, LitTxFMConfig  # noqa: E402
from txfm_repro.mock_data import mask_collate  # noqa: E402

HF_REPO = "gabrielaltay/tcga-patients-open"
HF_CONFIG = "TCGA-LUAD"
BATCH_SIZE = 4
K = 256
LIB_L = 1e5
CACHE_PATH = Path(__file__).resolve().parents[1] / "configs" / "gene_ids_gencode_v36.json"


class TCGAStreamPrototype(IterableDataset):
    def __init__(self, max_rows: int) -> None:
        self.max_rows = max_rows

    def __iter__(self):
        ds = load_dataset(HF_REPO, HF_CONFIG, split="train", streaming=True)
        for row in itertools.islice(ds, self.max_rows):
            aliquots = row.get("samples_gene_expression_quantification") or []
            if not aliquots:
                continue
            a = aliquots[0]
            counts = torch.tensor(a["unstranded"], dtype=torch.int64)
            mask = torch.ones_like(counts, dtype=torch.bool)
            yield counts, mask


def main() -> None:
    with CACHE_PATH.open() as f:
        gene_ids = json.load(f)
    G = len(gene_ids)
    print(f"n_genes (frozen GENCODE v36): {G}")

    proto = TCGAStreamPrototype(max_rows=BATCH_SIZE)
    loader = DataLoader(
        proto,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=functools.partial(mask_collate, K=K, library_size_L=LIB_L),
    )

    batch = next(iter(loader))
    print("\nbatch keys:")
    for k, v in batch.items():
        print(f"  {k:<14} shape={tuple(v.shape)}  dtype={v.dtype}")

    assert batch["unmasked_idx"].shape == (BATCH_SIZE, K)
    assert batch["target"].shape == (BATCH_SIZE, G)
    assert batch["target_mask"].all(), "TCGA aliquot should be full coverage"
    assert (batch["library_size"] > 0).all()
    print("\nlibrary_size:", batch["library_size"].tolist())

    cfg = LitTxFMConfig(n_genes=G, d_model=64, n_layers=1, n_heads=4, dim_ff=128,
                        decoder_layers=1, library_size_L=LIB_L)
    lit = LitTxFM(cfg)
    lit.train()
    loss = lit.training_step(batch, 0)
    print(f"\nLitTxFM.training_step ran; loss={loss.item():.4f}  (finite={torch.isfinite(loss).item()})")
    assert torch.isfinite(loss)
    print("OK")


if __name__ == "__main__":
    main()

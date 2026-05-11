"""Phase 1 sandbox: prototype an IterableDataset on top of the HF stream.

Wraps `load_dataset(..., streaming=True)`, skips rows with no expression,
picks the first aliquot, and yields `(counts: int64 (G,), measured_mask: bool (G,))`
— the same per-item shape that `MockBulkRNADataset.__getitem__` produces, so
the existing `mask_collate` works without modification.

Run: `uv run python dev_sandbox/03_iterable_dataset_prototype.py`
"""

from __future__ import annotations

import itertools

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

HF_REPO = "gabrielaltay/tcga-patients-open"
HF_CONFIG = "TCGA-LUAD"
N_ITEMS_TO_PULL = 5


class TCGAStreamPrototype(IterableDataset):
    def __init__(self, hf_repo: str, hf_config: str, max_rows: int | None = None) -> None:
        self.hf_repo = hf_repo
        self.hf_config = hf_config
        self.max_rows = max_rows

    def __iter__(self):
        ds = load_dataset(self.hf_repo, self.hf_config, split="train", streaming=True)
        if self.max_rows is not None:
            ds = itertools.islice(ds, self.max_rows)
        for row in ds:
            aliquots = row.get("samples_gene_expression_quantification") or []
            if not aliquots:
                continue
            a = aliquots[0]
            counts = torch.tensor(a["unstranded"], dtype=torch.int64)
            mask = torch.ones_like(counts, dtype=torch.bool)
            yield counts, mask


def main() -> None:
    proto = TCGAStreamPrototype(HF_REPO, HF_CONFIG, max_rows=N_ITEMS_TO_PULL)
    items = list(proto)
    print(f"pulled {len(items)} items")
    for i, (counts, mask) in enumerate(items):
        lib = int(counts.sum().item())
        nz = int((counts > 0).sum().item())
        print(
            f"  [{i}] counts: shape={tuple(counts.shape)} dtype={counts.dtype}  "
            f"sum={lib:,}  nonzero={nz}  mask.all={bool(mask.all())}"
        )
        assert counts.dtype == torch.int64
        assert mask.dtype == torch.bool
        assert counts.shape == mask.shape
        assert lib > 0
    print("OK")


if __name__ == "__main__":
    main()

"""Streaming TCGA RNA-seq data module.

Streams the public HuggingFace dataset `gabrielaltay/tcga-patients-open`
without ever materializing the full shard on disk. Each row is a patient;
the `samples_gene_expression_quantification` column is a list of aliquots
(one per RNA-seq run). We pick the first aliquot per patient, pull raw
`unstranded` counts, and emit `(counts, measured_mask)` pairs shaped like
`MockBulkRNADataset.__getitem__` — so the existing `mask_collate` does the
library-size norm + log1p + K-subselect step unchanged.

Train/val split is patient-level via a content-based hash bucket on
`case_id`: no pre-pass over the stream, no leakage by construction,
deterministic across runs.

Phase 1 limitations (documented; not bugs):
  - `aliquot_strategy="first"` only — "random" and "all" stubbed for Phase 2.
  - `num_workers=0` only. Multiple workers would each replay the same stream
    unless we shard via `worker_info`; that's a Phase 2 ticket.
  - No per-epoch shuffle. HF stream order is the natural patient order, which
    is fine for small projects (LUAD has ~400 patients).
"""

from __future__ import annotations

import functools
import hashlib
import json
import warnings
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from txfm_repro.mock_data import mask_collate

AliquotStrategy = Literal["first", "random", "all"]
Split = Literal["train", "val"]


def _patient_hash_bucket(case_id: str, n_buckets: int) -> int:
    """Deterministic content-based bucket assignment for patient-level splits.

    blake2b is used (not Python's `hash()`) because Python hashing is
    randomized per-process, so the same `case_id` would land in different
    buckets on different runs.
    """
    digest = hashlib.blake2b(case_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_buckets


def freeze_gene_universe(
    hf_repo: str,
    hf_config: str,
    cache_path: Path | str | None = None,
) -> list[str]:
    """Return the GENCODE v36 gene_id list, in the canonical order used by
    every aliquot in the HF dataset.

    Caching is opt-in but strongly recommended. If `cache_path` points at an
    existing JSON file, load and return its contents. Otherwise stream the
    first non-empty aliquot from HF, return its `gene_id`, and (if
    `cache_path` is set) write it to disk.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with cache_path.open() as f:
                return list(json.load(f))

    from datasets import load_dataset

    ds = load_dataset(hf_repo, hf_config, split="train", streaming=True)
    for row in ds:
        aliquots = row.get("samples_gene_expression_quantification") or []
        if not aliquots:
            continue
        gene_ids = list(aliquots[0]["gene_id"])
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w") as f:
                json.dump(gene_ids, f)
        return gene_ids
    raise RuntimeError(f"no aliquots found in {hf_repo} / {hf_config}")


class TCGAStreamingIterableDataset(IterableDataset):
    """Wraps an HF streaming dataset, yields per-patient `(counts, mask)` items.

    `_source_iterable_factory` is a test hook: when provided, it replaces the
    HF call entirely. The factory must return a fresh iterable of dicts each
    time it's invoked (HF IterableDatasets are single-pass, and Lightning
    will re-iterate per epoch).
    """

    def __init__(
        self,
        hf_repo: str,
        hf_config: str,
        gene_id_list: list[str],
        split: Split,
        val_hash_bucket: int = 0,
        n_hash_buckets: int = 10,
        aliquot_strategy: AliquotStrategy = "first",
        max_rows: int | None = None,
        with_case_id: bool = False,
        _source_iterable_factory: Callable[[], Iterable[dict]] | None = None,
    ) -> None:
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be 'train', 'val', or 'all', got {split!r}")
        if aliquot_strategy != "first":
            raise NotImplementedError(
                f"aliquot_strategy={aliquot_strategy!r} is a Phase 2 feature; "
                "only 'first' is implemented in Phase 1"
            )
        if not (0 <= val_hash_bucket < n_hash_buckets):
            raise ValueError(
                f"val_hash_bucket must be in [0, {n_hash_buckets}), got {val_hash_bucket}"
            )
        self.hf_repo = hf_repo
        self.hf_config = hf_config
        self.gene_id_list = gene_id_list
        self.split = split
        self.val_hash_bucket = val_hash_bucket
        self.n_hash_buckets = n_hash_buckets
        self.aliquot_strategy = aliquot_strategy
        self.max_rows = max_rows
        self.with_case_id = with_case_id
        self._source_iterable_factory = _source_iterable_factory

    def _source(self) -> Iterable[dict]:
        if self._source_iterable_factory is not None:
            return self._source_iterable_factory()
        from datasets import load_dataset

        return load_dataset(self.hf_repo, self.hf_config, split="train", streaming=True)

    def _row_in_split(self, case_id: str) -> bool:
        if self.split == "all":
            return True
        bucket = _patient_hash_bucket(case_id, self.n_hash_buckets)
        is_val = bucket == self.val_hash_bucket
        return is_val if self.split == "val" else not is_val

    def __iter__(self) -> Iterator:
        n_yielded = 0
        for row in self._source():
            if self.max_rows is not None and n_yielded >= self.max_rows:
                break
            aliquots = row.get("samples_gene_expression_quantification") or []
            if not aliquots:
                continue
            case_id = row.get("case_id")
            if case_id is None or not self._row_in_split(case_id):
                continue
            a = aliquots[0]
            counts = torch.tensor(a["unstranded"], dtype=torch.int64)
            # TCGA aliquots have full GENCODE v36 coverage. We still emit a
            # mask so the downstream collate sees the same per-row shape it
            # gets from MockBulkRNADataset (multi-source training mixes
            # full and partial coverage).
            mask = torch.ones_like(counts, dtype=torch.bool)
            if self.with_case_id:
                yield counts, mask, case_id
            else:
                yield counts, mask
            n_yielded += 1


class TCGADataModule(L.LightningDataModule):
    """Streaming TCGA RNA-seq DataModule.

    The flat init signature mirrors `MockBulkDataModule` so Lightning CLI
    exposes every knob as a `data.init_args.*` argument and YAML key.
    """

    def __init__(
        self,
        hf_repo: str = "gabrielaltay/tcga-patients-open",
        hf_config: str = "TCGA-LUAD",
        n_genes: int = 60660,
        gene_id_cache_path: str = "configs/gene_ids_gencode_v36.json",
        K_unmasked: int = 1024,
        K_unmasked_min: int | None = None,
        library_size_L: float = 1e5,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 0,
        val_hash_bucket: int = 0,
        n_hash_buckets: int = 10,
        aliquot_strategy: AliquotStrategy = "first",
        max_train_rows: int | None = None,
        max_val_rows: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if num_workers != 0:
            warnings.warn(
                "TCGADataModule with num_workers>0 will replay the same stream "
                "in each worker (Phase 1 limitation). Use 0 until per-worker "
                "row sharding via worker_info lands in Phase 2.",
                stacklevel=2,
            )
        self.hf_repo = hf_repo
        self.hf_config = hf_config
        self.n_genes = n_genes
        self.gene_id_cache_path = gene_id_cache_path
        self.K_unmasked = K_unmasked
        self.K_unmasked_min = K_unmasked_min
        self.library_size_L = library_size_L
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_hash_bucket = val_hash_bucket
        self.n_hash_buckets = n_hash_buckets
        self.aliquot_strategy = aliquot_strategy
        self.max_train_rows = max_train_rows
        self.max_val_rows = max_val_rows

        self.gene_id_list: list[str] | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.gene_id_list is None:
            self.gene_id_list = freeze_gene_universe(
                self.hf_repo, self.hf_config, cache_path=self.gene_id_cache_path,
            )
        if len(self.gene_id_list) != self.n_genes:
            raise ValueError(
                f"frozen gene universe has {len(self.gene_id_list)} entries but "
                f"config declares n_genes={self.n_genes}. Update the YAML."
            )

    def _make_dataset(self, split: Split, max_rows: int | None) -> TCGAStreamingIterableDataset:
        assert self.gene_id_list is not None, "call setup() first"
        return TCGAStreamingIterableDataset(
            hf_repo=self.hf_repo,
            hf_config=self.hf_config,
            gene_id_list=self.gene_id_list,
            split=split,
            val_hash_bucket=self.val_hash_bucket,
            n_hash_buckets=self.n_hash_buckets,
            aliquot_strategy=self.aliquot_strategy,
            max_rows=max_rows,
        )

    def _collate(self):
        return functools.partial(
            mask_collate,
            K=self.K_unmasked,
            library_size_L=self.library_size_L,
            K_min=self.K_unmasked_min,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make_dataset("train", self.max_train_rows),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make_dataset("val", self.max_val_rows),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
        )

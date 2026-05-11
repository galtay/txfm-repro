"""Phase 1 sandbox: prove the gene_id ordering is invariant across aliquots/patients.

Streams the first ~20 LUAD patients, hashes each aliquot's gene_id tuple,
and asserts a single distinct hash. If so, dumps the canonical gene_id list
to `configs/gene_ids_gencode_v36.json` so production code can avoid the
network round-trip.

Run: `uv run python dev_sandbox/02_gene_universe_check.py`
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from datasets import load_dataset

HF_REPO = "gabrielaltay/tcga-patients-open"
HF_CONFIG = "TCGA-LUAD"
N_PATIENTS_TO_PROBE = 20
CACHE_PATH = Path(__file__).resolve().parents[1] / "configs" / "gene_ids_gencode_v36.json"


def main() -> None:
    ds = load_dataset(HF_REPO, HF_CONFIG, split="train", streaming=True)

    distinct_hashes: set[int] = set()
    canonical: list[str] | None = None
    n_aliquots_seen = 0

    for row in itertools.islice(ds, N_PATIENTS_TO_PROBE):
        aliquots = row.get("samples_gene_expression_quantification") or []
        for a in aliquots:
            gene_ids = tuple(a["gene_id"])
            distinct_hashes.add(hash(gene_ids))
            if canonical is None:
                canonical = list(gene_ids)
            n_aliquots_seen += 1

    if canonical is None:
        raise RuntimeError("no aliquots found in first 20 patients — dataset broken?")

    print(f"probed {N_PATIENTS_TO_PROBE} patients, {n_aliquots_seen} aliquots")
    print(f"distinct gene_id orderings: {len(distinct_hashes)}")
    assert len(distinct_hashes) == 1, "gene order is NOT invariant — design assumption broken"
    print(f"len(gene_id)={len(canonical)}")
    print(f"first 5: {canonical[:5]}")
    print(f"last 5:  {canonical[-5:]}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w") as f:
        json.dump(canonical, f)
    print(f"\nwrote {CACHE_PATH} ({CACHE_PATH.stat().st_size:,} bytes)")
    print("OK")


if __name__ == "__main__":
    main()

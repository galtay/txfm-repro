"""Phase 1 sandbox: probe the gabrielaltay/tcga-patients-open HF dataset.

Streams TCGA-LUAD without downloading the full shard. Pulls the first row,
inspects the schema of `samples_gene_expression_quantification`, and prints
a few sanity-check numbers (gene_id length, library-size sum).

Run: `uv run python dev_sandbox/01_explore_hf_stream.py`
"""

from __future__ import annotations

from datasets import load_dataset

HF_REPO = "gabrielaltay/tcga-patients-open"
HF_CONFIG = "TCGA-LUAD"


def main() -> None:
    ds = load_dataset(HF_REPO, HF_CONFIG, split="train", streaming=True)
    it = iter(ds)
    row = next(it)

    print("top-level keys:")
    for k in row:
        print(f"  - {k}")

    aliquots = row.get("samples_gene_expression_quantification") or []
    print(f"\ncase_id={row.get('case_id')}  project_id={row.get('project_id')}")
    print(f"n_aliquots={len(aliquots)}")
    if not aliquots:
        print("first row has no expression aliquots — try a later row")
        return

    a = aliquots[0]
    print("\naliquot keys:")
    for k in a:
        print(f"  - {k}")

    n_genes = len(a["gene_id"])
    lib = sum(a["unstranded"])
    print(f"\nlen(gene_id)={n_genes}")
    print(f"gene_id[:3]={a['gene_id'][:3]}")
    print(f"gene_id[-3:]={a['gene_id'][-3:]}")
    print(f"unstranded[:5]={a['unstranded'][:5]}")
    print(f"sum(unstranded)={lib:,}")
    assert lib > 1e6, f"library size {lib} is suspiciously small"
    assert n_genes > 50_000, f"only {n_genes} genes — expected ~60660 (GENCODE v36)"
    print("\nOK")


if __name__ == "__main__":
    main()

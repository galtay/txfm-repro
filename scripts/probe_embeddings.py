"""Downstream probes on extracted TCGA CLS embeddings.

Loads the parquet emitted by `scripts/extract_embeddings.py` and runs:

  1. `sample_type` linear probe — logistic regression on the embedding,
     trained on the train-bucket patients and evaluated on the val bucket.
     Reports accuracy + balanced accuracy + the majority-class baseline.
  2. Survival concordance index — fits a Ridge regression on `os_time`
     (using `os_event=1` patients as direct supervision, masking out the
     censored ones from training but scoring everyone). The predicted
     time is converted to a risk score (`-predicted_time`) and the
     concordance index is computed across all comparable pairs.

Outputs a JSON to `--out` and prints a brief prose summary. Each probe
is independent — if `sample_type` has only one class in val or no
`os_event=1` patients exist, that probe is skipped with a reason.

Run:
  uv run python scripts/probe_embeddings.py \\
      --embeddings out/embeddings_luad.parquet \\
      --out out/probe_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def _emb_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("embedding_")],
        key=lambda c: int(c.split("_")[1]),
    )


def concordance_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    """Standard C-index: over all comparable pairs (where the earlier-event
    sample's time is the smaller of two times and that sample experienced
    the event), fraction where the model assigns higher risk to the
    earlier-event sample.
    """
    n = len(times)
    num = 0.0
    denom = 0.0
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j or times[j] <= times[i]:
                continue
            denom += 1.0
            if risks[i] > risks[j]:
                num += 1.0
            elif risks[i] == risks[j]:
                num += 0.5
    return float(num / denom) if denom > 0 else float("nan")


def probe_sample_type(df: pd.DataFrame, val_bucket: int) -> dict:
    if "sample_type" not in df or "hash_bucket" not in df:
        return {"skipped": "missing sample_type / hash_bucket columns"}
    sub = df.dropna(subset=["sample_type"]).copy()
    if sub.empty:
        return {"skipped": "no rows with sample_type"}
    train = sub[sub["hash_bucket"] != val_bucket]
    val = sub[sub["hash_bucket"] == val_bucket]
    if train["sample_type"].nunique() < 2:
        return {"skipped": f"only {train['sample_type'].nunique()} sample_type class in train split"}
    if val.empty:
        return {"skipped": "val split is empty"}

    emb_cols = _emb_columns(sub)
    X_train = train[emb_cols].to_numpy()
    y_train = train["sample_type"].to_numpy()
    X_val = val[emb_cols].to_numpy()
    y_val = val["sample_type"].to_numpy()

    clf = LogisticRegressionCV(max_iter=2000, class_weight="balanced", cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    val_modal = pd.Series(y_val).mode().iloc[0]
    majority_acc = float((y_val == val_modal).mean())

    return {
        "n_train":            int(len(train)),
        "n_val":              int(len(val)),
        "classes":            sorted(map(str, train["sample_type"].unique())),
        "accuracy":           float(accuracy_score(y_val, y_pred)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_val, y_pred)),
        "majority_baseline":  majority_acc,
    }


def probe_survival(df: pd.DataFrame) -> dict:
    if not {"os_event", "os_time"}.issubset(df.columns):
        return {"skipped": "missing os_event / os_time columns"}
    sub = df.dropna(subset=["os_event", "os_time"]).copy()
    sub["os_event"] = sub["os_event"].astype(int)
    sub["os_time"] = sub["os_time"].astype(float)
    sub = sub[sub["os_time"] > 0]
    if (sub["os_event"] == 1).sum() < 2:
        return {"skipped": "fewer than 2 patients with os_event=1"}

    emb_cols = _emb_columns(sub)
    # Use only event-true patients as training supervision (avoids censor bias).
    # Score everyone.
    train = sub[sub["os_event"] == 1]
    X_train = train[emb_cols].to_numpy()
    y_train = train["os_time"].to_numpy()
    X_all = sub[emb_cols].to_numpy()

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    predicted_time = reg.predict(X_all)
    risk = -predicted_time  # shorter predicted survival → higher risk

    cidx = concordance_index(
        times=sub["os_time"].to_numpy(),
        events=sub["os_event"].to_numpy(),
        risks=risk,
    )
    return {
        "n_total":           int(len(sub)),
        "n_events":          int((sub["os_event"] == 1).sum()),
        "concordance_index": cidx,
        "note":              "trained ridge on event=1 patients' os_time, scored full cohort",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--val-hash-bucket", type=int, default=0)
    args = p.parse_args()

    df = pd.read_parquet(args.embeddings)
    print(f"loaded {len(df)} rows from {args.embeddings}")
    print(f"embedding dim: {len(_emb_columns(df))}")
    print(f"hash_bucket distribution: {df['hash_bucket'].value_counts().sort_index().to_dict()}")
    print()

    results = {
        "n_rows":          int(len(df)),
        "embedding_dim":   len(_emb_columns(df)),
        "sample_type":     probe_sample_type(df, args.val_hash_bucket),
        "survival":        probe_survival(df),
    }

    print("== sample_type probe ==")
    for k, v in results["sample_type"].items():
        print(f"  {k}: {v}")
    print()
    print("== survival concordance ==")
    for k, v in results["survival"].items():
        print(f"  {k}: {v}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Phase 1 sandbox: validate the patient-level hash split.

Hashes 200 fake `case_id`s using blake2b, mods by `n_buckets=10`, and asserts:
- The train/val splits are disjoint at the patient level.
- The val fraction lands close to 1/n_buckets (~10%).

Run: `uv run python dev_sandbox/05_patient_split_hash.py`
"""

from __future__ import annotations

import hashlib
import uuid


def patient_hash_bucket(case_id: str, n_buckets: int) -> int:
    digest = hashlib.blake2b(case_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_buckets


def main() -> None:
    n_buckets = 10
    val_bucket = 0
    case_ids = [str(uuid.uuid4()) for _ in range(200)]

    train, val = [], []
    for cid in case_ids:
        (val if patient_hash_bucket(cid, n_buckets) == val_bucket else train).append(cid)

    print(f"total={len(case_ids)}  train={len(train)}  val={len(val)}")
    assert set(train).isdisjoint(set(val)), "train/val overlap — split broken"

    val_frac = len(val) / len(case_ids)
    print(f"val_frac={val_frac:.3f}  (expected ~{1/n_buckets:.3f})")
    assert 0.04 < val_frac < 0.18, f"val_frac={val_frac} far from 1/{n_buckets}"

    # Deterministic — same case_id always lands in same bucket.
    cid = case_ids[0]
    assert patient_hash_bucket(cid, n_buckets) == patient_hash_bucket(cid, n_buckets)
    print("OK")


if __name__ == "__main__":
    main()

"""Cross-check a cohort against SleepFM's official dataset_split.json.

Phase 15'in "clean cohort SleepFM pretrain'de değil" iddiasının kod
seviyesinde tekrarlanabilir kanıtı. Bulguları CSV olarak yazar.

Kullanım:
  python scripts/verify_cohort_pretrain_independence.py

Beklenen (Phase 13 clean cohort için):
  pretrain overlap: 0
  train overlap: 0
  validation overlap: 0
  test overlap: 20  (tam eşleşme)

Bu script her yeni cohort için (Phase 16, n=100 vs.) tekrar koşulmalı ve
çıktı reports/pretrain_independence/ altında commit edilmelidir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


# Known-good SHA256 of the SleepFM upstream `dataset_split.json` this cohort
# was validated against. If the upstream repo updates the manifest, this
# hash should change and the analysis must be re-audited. The script warns
# on mismatch but does not fail — new upstream splits may be legitimate.
KNOWN_SPLIT_SHA256 = "57d5019a0d67a1209fe1f8caa4a5f96067dec0725ac7a79e04fe407201a577db"


import os

_REPO = Path(__file__).resolve().parents[1]


def _resolve_split_file() -> Path:
    """Locate dataset_split.json from env var or common sibling repo layouts."""
    env = os.environ.get("SLEEPFM_UPSTREAM_DIR", "").strip()
    if env:
        return Path(env) / "sleepfm" / "configs" / "dataset_split.json"
    for cand in [
        _REPO.parent / "SleepFM" / "sleepFMoriginal" / "sleepfm-clinical" / "sleepfm" / "configs" / "dataset_split.json",
        _REPO.parent / "sleepfm-clinical" / "sleepfm" / "configs" / "dataset_split.json",
    ]:
        if cand.exists():
            return cand
    # Historical default (developer machine)
    return Path("C:/Users/User/Desktop/Projeler/SleepFM/sleepFMoriginal/"
                "sleepfm-clinical/sleepfm/configs/dataset_split.json")


SPLIT_FILE_DEFAULT = _resolve_split_file()
COHORT_DIR_DEFAULT = _REPO / "data" / "clean_cohort_run" / "embeddings"
REPORT_DIR_DEFAULT = _REPO / "reports" / "pretrain_independence"

SUBJECT_RE = re.compile(r"mesa-sleep-(\d{4})")


def extract_ids(paths):
    ids = set()
    for p in paths:
        m = SUBJECT_RE.search(str(p))
        if m:
            ids.add(m.group(1))
    return ids


def cohort_ids_from_dir(cohort_dir: Path) -> set[str]:
    return extract_ids(cohort_dir.glob("mesa-sleep-*"))


def compute_sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_splits(split_file: Path) -> dict[str, set[str]]:
    with split_file.open() as fh:
        raw = json.load(fh)
    return {name: extract_ids(entries) for name, entries in raw.items()}


def crosscheck(cohort: set[str], splits: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for split_name in ["pretrain", "train", "validation", "test",
                       "temporal_test", "external_validation"]:
        split_ids = splits.get(split_name, set())
        overlap = sorted(cohort & split_ids)
        rows.append({
            "split": split_name,
            "split_size": len(split_ids),
            "cohort_size": len(cohort),
            "overlap_count": len(overlap),
            "overlap_ids": ",".join(overlap) if overlap else "",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE_DEFAULT)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT_DIR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR_DEFAULT)
    parser.add_argument("--cohort-label", type=str, default="clean_cohort_n20")
    args = parser.parse_args()

    if not args.split_file.exists():
        print(f"[error] Split file not found: {args.split_file}", file=sys.stderr)
        sys.exit(2)
    if not args.cohort_dir.exists():
        print(f"[error] Cohort dir not found: {args.cohort_dir}", file=sys.stderr)
        sys.exit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_sha = compute_sha256(args.split_file)
    if split_sha == KNOWN_SPLIT_SHA256:
        print(f"[ok] dataset_split.json SHA matches known-good: {split_sha[:12]}...")
    else:
        print(f"[warn] dataset_split.json SHA mismatch")
        print(f"       expected: {KNOWN_SPLIT_SHA256}")
        print(f"       got:      {split_sha}")
        print(f"       upstream repo may have updated the manifest; re-audit needed")

    cohort = cohort_ids_from_dir(args.cohort_dir)
    splits = load_splits(args.split_file)
    df = crosscheck(cohort, splits)
    df.insert(0, "split_file_sha256", split_sha)

    print(f"Cohort ({args.cohort_label}): {len(cohort)} subjects")
    print(f"IDs: {sorted(cohort)}\n")
    print(df.to_string(index=False))

    csv_path = args.out_dir / f"{args.cohort_label}_vs_sleepfm_splits.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    pretrain_overlap = df.loc[df["split"] == "pretrain", "overlap_count"].iloc[0]
    if pretrain_overlap > 0:
        print(f"\n[FAIL] {pretrain_overlap} cohort subjects in SleepFM pretrain")
        print("Phase 15 memorization argument INVALID for this cohort.")
        sys.exit(1)

    test_overlap = df.loc[df["split"] == "test", "overlap_count"].iloc[0]
    coverage = test_overlap / len(cohort) if cohort else 0
    print(f"\n[OK] pretrain overlap = 0; test overlap = {test_overlap}/{len(cohort)} "
          f"({100 * coverage:.0f}%)")
    print("Cohort defensibly held out of SleepFM pretraining.")


if __name__ == "__main__":
    main()

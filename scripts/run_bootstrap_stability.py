"""Phase 16 — bootstrap_stability across modalities on a clean cohort.

Subject-drop-out robustness of cluster count + noise fraction: dropping
some fraction of subjects and refitting UMAP+HDBSCAN measures how much
of the "5 clusters" claim depends on which subjects happened to be in
the cohort. Complements `multiseed_stability` (which only varies the
UMAP init seed on the full dataset).

Usage:
    python scripts/run_bootstrap_stability.py

Defaults reproduce the numbers in reports/phase15_defensibility/
bootstrap_stability_summary.csv. Override any argument for a different
cohort, output directory, or bootstrap budget:

    python scripts/run_bootstrap_stability.py \
        --cohort-dir data/n100_cohort_run/embeddings \
        --output-dir reports/phase17_defensibility \
        --n-bootstraps 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from real_embeddings import (  # noqa: E402
    load_subject_embeddings,
    load_subject_embeddings_multimodal,
)
from rigor_analysis import bootstrap_stability  # noqa: E402


DEFAULT_COHORT_DIR = REPO / "data" / "clean_cohort_run" / "embeddings"
DEFAULT_OUTPUT_DIR = REPO / "reports" / "phase15_defensibility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cohort-dir", type=Path, default=DEFAULT_COHORT_DIR,
        help=f"Directory of *_embeddings.hdf5 files (default: {DEFAULT_COHORT_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write summary CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--n-bootstraps", type=int, default=50,
        help="Bootstrap draws per modality (default: 50)",
    )
    parser.add_argument(
        "--subsample-frac", type=float, default=0.8,
        help="Fraction of subjects sampled per bootstrap (default: 0.8)",
    )
    parser.add_argument(
        "--umap-n-neighbors", type=int, default=5,
        help="UMAP n_neighbors (default: 5 — matches rigor_analysis)",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size", type=int, default=2,
        help="HDBSCAN min_cluster_size (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base RNG seed (default: 42)",
    )
    return parser.parse_args()


def run_modality(
    modality: str,
    X: np.ndarray,
    subject_ids: list[str],
    n_bootstraps: int,
    subsample_frac: float,
    umap_n_neighbors: int,
    hdbscan_min_cluster_size: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    df = bootstrap_stability(
        X,
        n_bootstraps=n_bootstraps,
        subsample_frac=subsample_frac,
        umap_n_neighbors=umap_n_neighbors,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        seed=seed,
    )
    df["modality"] = modality

    summary = {
        "modality": modality,
        "n_subjects_full": len(subject_ids),
        "subsample_size": int(subsample_frac * len(subject_ids)),
        "n_bootstraps": n_bootstraps,
        "clusters_mean": float(df["n_clusters"].mean()),
        "clusters_std": float(df["n_clusters"].std()),
        "clusters_min": int(df["n_clusters"].min()),
        "clusters_max": int(df["n_clusters"].max()),
        "clusters_cv": (
            float(df["n_clusters"].std() / df["n_clusters"].mean())
            if df["n_clusters"].mean() > 0 else float("nan")
        ),
        "noise_frac_mean": float(df["noise_fraction"].mean()),
        "noise_frac_std": float(df["noise_fraction"].std()),
    }
    return summary, df


def main() -> int:
    args = parse_args()

    if not args.cohort_dir.exists():
        print(f"[error] cohort dir not found: {args.cohort_dir}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"bootstrap_stability across modalities  ({args.cohort_dir.name})")
    print(f"  n_bootstraps={args.n_bootstraps}, subsample_frac={args.subsample_frac}")
    print("=" * 72)

    summary_rows = []
    per_bootstrap = []

    for modality in ["BAS", "RESP", "EKG", "EMG"]:
        print(f"\n--- {modality} ---")
        X, subject_ids = load_subject_embeddings(
            embedding_dir=args.cohort_dir, modality=modality,
        )
        print(f"  loaded X shape={X.shape}, subjects={len(subject_ids)}")
        summary, df = run_modality(
            modality, X, subject_ids,
            args.n_bootstraps, args.subsample_frac,
            args.umap_n_neighbors, args.hdbscan_min_cluster_size, args.seed,
        )
        summary_rows.append(summary)
        per_bootstrap.append(df)
        print(f"  clusters: mean={summary['clusters_mean']:.2f} "
              f"std={summary['clusters_std']:.2f} "
              f"range=[{summary['clusters_min']}, {summary['clusters_max']}]")
        print(f"  noise_frac: mean={summary['noise_frac_mean']:.3f}")

    print(f"\n--- MULTI (4-modality concat, 512-dim) ---")
    X_multi, sids_multi, skipped = load_subject_embeddings_multimodal(
        embedding_dir=args.cohort_dir,
    )
    print(f"  loaded X shape={X_multi.shape}, subjects={len(sids_multi)}")
    summary, df_multi = run_modality(
        "MULTI", X_multi, sids_multi,
        args.n_bootstraps, args.subsample_frac,
        args.umap_n_neighbors, args.hdbscan_min_cluster_size, args.seed,
    )
    summary_rows.append(summary)
    per_bootstrap.append(df_multi)
    print(f"  clusters: mean={summary['clusters_mean']:.2f} "
          f"std={summary['clusters_std']:.2f}")

    summary_df = pd.DataFrame(summary_rows)
    per_boot_df = pd.concat(per_bootstrap, ignore_index=True)

    summary_csv = args.output_dir / "bootstrap_stability_summary.csv"
    per_boot_csv = args.output_dir / "bootstrap_stability_per_bootstrap.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_boot_df.to_csv(per_boot_csv, index=False)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print(f"\n[save] {summary_csv}")
    print(f"[save] {per_boot_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

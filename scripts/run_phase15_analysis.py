"""Phase 15 defensibility analysis — chance baseline + cross-modality FWER
+ bootstrap CI + purity Δ.

Denetim TIER 2: bu script Phase 15'in headline sayılarını (subject purity
null baselines, cross-modality FWER, ARI bootstrap CI, purity Δ) üretir.
Phase 16r'ye kadar scratchpad'te yaşıyordu; artık repo'nun kalıcı parçası.

Usage:
    python scripts/run_phase15_analysis.py

Defaults reproduce the committed reports/phase15_defensibility/*.csv
files. Override for a different cohort:

    python scripts/run_phase15_analysis.py \
        --output-dir reports/phase17_defensibility \
        --phase13c-dir reports/phase17_rigor_clean \
        --hdbscan-min-cluster-size 30 \
        --n-permutations 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chunk_level_analysis import run_chunk_analysis  # noqa: E402
from rigor_analysis import (  # noqa: E402
    bonferroni_fdr_family,
    bootstrap_ci_mean,
    null_purity_from_label_shuffle,
    session_shuffle_null_purity,
)


DEFAULT_OUTPUT = REPO / "reports" / "phase15_defensibility"
DEFAULT_PHASE13C = REPO / "reports" / "phase13c_rigor_clean"

# Contamined-cohort purity values from Phase 8e / 9e reports (used as a
# fixed reference; not recomputed). If regenerating for a different
# reference, edit these or make them CLI args.
CONTAMINATED_PURITY = {
    "BAS": 0.74,
    "RESP": 0.37,
    "EKG": 0.55,
    "EMG": 0.48,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--phase13c-dir", type=Path, default=DEFAULT_PHASE13C,
                   help="Directory with correction_*.csv, stability_*.csv, "
                        "stability_ari_matrix_*.npy from Phase 13c-style rigor pass")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    p.add_argument("--n-permutations", type=int, default=500,
                   help="Label-shuffle permutations. Note: p-value floor is "
                        "1/(n+1); raise for tighter bounds.")
    p.add_argument("--n-session-permutations", type=int, default=200)
    p.add_argument("--n-bootstraps", type=int, default=2000)
    return p.parse_args()


def compute_purity_null_baselines(
    modalities: list[str], hdbscan_mcs: int, n_permutations: int,
    n_session_permutations: int,
) -> pd.DataFrame:
    rows = []
    for modality in modalities:
        print(f"\n--- {modality} ---")
        result = run_chunk_analysis(
            modality=modality, hdbscan_min_cluster_size=hdbscan_mcs,
        )
        obs, null_mean, p_ls, _ = null_purity_from_label_shuffle(
            result.cluster_labels, result.subject_ids,
            n_permutations=n_permutations,
        )
        obs_ss, null_mean_ss, p_ss = session_shuffle_null_purity(
            result.cluster_labels, result.subject_ids,
            n_permutations=n_session_permutations,
        )
        print(f"  Observed subject purity: {obs:.3f}")
        print(f"  Label-shuffle null mean: {null_mean:.3f}  p={p_ls:.4f}")
        print(f"  Session-shuffle null mean: {null_mean_ss:.3f}  p={p_ss:.4f}")
        n_clust = int(len(np.unique(
            result.cluster_labels[result.cluster_labels >= 0]
        )))
        rows.append({
            "modality": modality,
            "n_chunks": len(result.cluster_labels),
            "n_clusters": n_clust,
            "observed_purity": obs,
            "label_shuffle_null_mean": null_mean,
            "label_shuffle_p": p_ls,
            "session_shuffle_null_mean": null_mean_ss,
            "session_shuffle_p": p_ss,
        })
    return pd.DataFrame(rows)


def compute_cross_modality_fwer(phase13c_dir: Path) -> pd.DataFrame:
    p_by_test = {}
    for mod in ["BAS", "RESP", "EKG", "EMG", "MULTI"]:
        csv = phase13c_dir / f"correction_{mod}.csv"
        if not csv.exists():
            print(f"  [skip] {csv.name} not found")
            continue
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            p_by_test[(mod, row["variable"])] = float(row["raw_p"])
    print(f"\nTotal tests in cross-modality family: {len(p_by_test)}")
    return bonferroni_fdr_family(p_by_test, alpha=0.05)


def compute_ari_bootstrap_ci(
    phase13c_dir: Path, n_bootstraps: int,
) -> pd.DataFrame:
    rows = []
    for mod in ["BAS", "RESP", "EKG", "EMG", "MULTI"]:
        matrix_path = phase13c_dir / f"stability_ari_matrix_{mod}.npy"
        if not matrix_path.exists():
            print(f"  [skip] {matrix_path.name} not found")
            continue
        ari_matrix = np.load(matrix_path)
        upper = ari_matrix[np.triu_indices(ari_matrix.shape[0], k=1)]
        mean, lo, hi = bootstrap_ci_mean(upper, n_bootstraps=n_bootstraps)
        rows.append({
            "modality": mod, "mean_ari": mean,
            "ci95_low": lo, "ci95_high": hi,
        })
    return pd.DataFrame(rows)


def compute_purity_delta(purity_df: pd.DataFrame) -> pd.DataFrame:
    comparison = pd.DataFrame([
        {
            "modality": mod,
            "purity_contaminated": CONTAMINATED_PURITY[mod],
            "purity_clean": purity_df.set_index("modality").loc[mod, "observed_purity"],
        }
        for mod in ["BAS", "RESP", "EKG", "EMG"]
    ])
    comparison["delta"] = comparison["purity_contaminated"] - comparison["purity_clean"]
    return comparison.merge(
        purity_df[["modality", "label_shuffle_null_mean", "label_shuffle_p"]],
        on="modality",
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.phase13c_dir.exists():
        print(f"[error] phase13c-dir not found: {args.phase13c_dir}",
              file=sys.stderr)
        return 2

    print("=" * 72)
    print("Phase 15 defensibility analysis")
    print(f"  output-dir     : {args.output_dir}")
    print(f"  phase13c-dir   : {args.phase13c_dir}")
    print(f"  hdbscan_mcs    : {args.hdbscan_min_cluster_size}")
    print(f"  n_permutations : {args.n_permutations}")
    print("=" * 72)

    print("\n[1/4] Chance baseline — subject purity per modality")
    purity_df = compute_purity_null_baselines(
        ["BAS", "RESP", "EKG", "EMG"],
        args.hdbscan_min_cluster_size,
        args.n_permutations,
        args.n_session_permutations,
    )
    purity_csv = args.output_dir / "subject_purity_null_baselines.csv"
    purity_df.to_csv(purity_csv, index=False)
    print(f"[save] {purity_csv}")

    print("\n[2/4] Cross-modality FWER over Phase 13c correction CSVs")
    family_df = compute_cross_modality_fwer(args.phase13c_dir)
    n_bonf = int(family_df["reject_bonferroni"].sum())
    n_fdr = int(family_df["reject_fdr"].sum())
    print(f"  Bonferroni-significant across modalities: {n_bonf}")
    print(f"  FDR-significant across modalities: {n_fdr}")
    family_csv = args.output_dir / "cross_modality_family_correction.csv"
    family_df.to_csv(family_csv, index=False)
    print(f"[save] {family_csv}")

    print("\n[3/4] ARI bootstrap CI over 10-seed stability matrices")
    ari_df = compute_ari_bootstrap_ci(args.phase13c_dir, args.n_bootstraps)
    print(ari_df.to_string(index=False))
    ari_csv = args.output_dir / "ari_bootstrap_ci.csv"
    ari_df.to_csv(ari_csv, index=False)
    print(f"[save] {ari_csv}")

    print("\n[4/4] Contaminated vs clean purity Δ")
    delta_df = compute_purity_delta(purity_df)
    print(delta_df.to_string(index=False))
    delta_csv = args.output_dir / "purity_contaminated_vs_clean.csv"
    delta_df.to_csv(delta_csv, index=False)
    print(f"[save] {delta_csv}")

    print(f"\n[done] Reports in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

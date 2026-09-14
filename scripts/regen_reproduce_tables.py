"""Regenerate the numeric snippets embedded in REPRODUCE.md §9-11 directly
from the committed CSV outputs.

Purpose: REPRODUCE.md carries "beklenen sayılar" that must match the
CSVs Phase 15/16 produced. Copying by hand drifts. Run this after any
report regeneration and paste the printed sections into REPRODUCE.md.

Usage:
  python scripts/regen_reproduce_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PHASE13C = REPO / "reports" / "phase13c_rigor_clean"
PHASE15 = REPO / "reports" / "phase15_defensibility"


def fmt_summary_headline() -> str:
    df = pd.read_csv(PHASE13C / "SUMMARY.csv")
    lines = ["## §9 REPRODUCE headline (Phase 13c) — birebir committed CSV\n"]
    for _, row in df.iterrows():
        lines.append(
            f"- **{row['modality']}**: min_p={row['min_per_var_p']:.4f}, "
            f"familywise min-p={row['min_familywise_p']:.4f}, "
            f"ARI={row['mean_ari']:.3f} "
            f"(std={row['std_ari']:.3f}, n_clusters_mode={row['n_clusters_mode']})"
        )
    return "\n".join(lines)


def fmt_purity_null() -> str:
    df = pd.read_csv(PHASE15 / "subject_purity_null_baselines.csv")
    lines = ["## §11 subject_purity_null_baselines.csv — birebir committed\n"]
    lines.append("```")
    lines.append("modality  n_chunks  n_clusters  observed_purity  label_shuffle_null_mean  label_shuffle_p")
    for _, row in df.iterrows():
        lines.append(
            f"{row['modality']:>8}  "
            f"{int(row['n_chunks']):>8}  "
            f"{int(row['n_clusters']):>10}  "
            f"{row['observed_purity']:>15.6f}  "
            f"{row['label_shuffle_null_mean']:>23.6f}  "
            f"{row['label_shuffle_p']:>15.6f}"
        )
    lines.append("```")
    return "\n".join(lines)


def fmt_ari_ci() -> str:
    df = pd.read_csv(PHASE15 / "ari_bootstrap_ci.csv")
    lines = ["## §10 ari_bootstrap_ci.csv — birebir committed\n"]
    lines.append("```")
    lines.append("modality  mean_ari  ci95_low  ci95_high")
    for _, row in df.iterrows():
        lines.append(
            f"{row['modality']:>8}  "
            f"{row['mean_ari']:>8.4f}  "
            f"{row['ci95_low']:>8.4f}  "
            f"{row['ci95_high']:>9.4f}"
        )
    lines.append("```")
    return "\n".join(lines)


def fmt_bootstrap_stability() -> str:
    df = pd.read_csv(PHASE15 / "bootstrap_stability_summary.csv")
    lines = ["## §10 bootstrap_stability_summary.csv — birebir committed\n"]
    lines.append("```")
    lines.append("modality  clusters_mean  clusters_std  clusters_min  clusters_max  clusters_cv")
    for _, row in df.iterrows():
        lines.append(
            f"{row['modality']:>8}  "
            f"{row['clusters_mean']:>13.2f}  "
            f"{row['clusters_std']:>12.3f}  "
            f"{int(row['clusters_min']):>12}  "
            f"{int(row['clusters_max']):>12}  "
            f"{row['clusters_cv'] * 100:>10.1f}%"
        )
    lines.append("```")
    return "\n".join(lines)


def main():
    print("=" * 72)
    print("REPRODUCE.md numeric snippet regeneration")
    print("Paste each section into the matching § in REPRODUCE.md")
    print("=" * 72)
    print()
    print(fmt_summary_headline())
    print()
    print(fmt_ari_ci())
    print()
    print(fmt_bootstrap_stability())
    print()
    print(fmt_purity_null())


if __name__ == "__main__":
    main()

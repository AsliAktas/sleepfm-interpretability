"""Phase 16 — bootstrap_stability across modalities on clean cohort (n=20).

Denetçinin TIER 2 bulgusu: `bootstrap_stability` kod var ama Phase 15 için
koşulmadı. Subject-drop-out robustness cluster count + noise fraction'ın
subsample varyansını ölçer — multiseed_stability sadece UMAP init varyansı
ölçtüğü için bu ayrı bir güvence katmanıdır.
"""
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/User/Desktop/Projeler/sleepfm_interpretability/src")

import numpy as np
import pandas as pd

from real_embeddings import (
    load_subject_embeddings,
    load_subject_embeddings_multimodal,
)
from rigor_analysis import bootstrap_stability


OUT = Path("C:/Users/User/Desktop/Projeler/sleepfm_interpretability/reports/phase15_defensibility")
OUT.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAPS = 50
SUBSAMPLE_FRAC = 0.8  # 20 × 0.8 = 16 subjects per bootstrap

print("=" * 72)
print("Phase 16 — bootstrap_stability across modalities (clean cohort, n=20)")
print(f"  n_bootstraps={N_BOOTSTRAPS}, subsample_frac={SUBSAMPLE_FRAC}")
print("=" * 72)

summary_rows = []
per_bootstrap = []

for modality in ["BAS", "RESP", "EKG", "EMG"]:
    print(f"\n--- {modality} ---")
    X, subject_ids = load_subject_embeddings(modality=modality)
    print(f"  loaded X shape={X.shape}, subjects={len(subject_ids)}")

    df = bootstrap_stability(
        X,
        n_bootstraps=N_BOOTSTRAPS,
        subsample_frac=SUBSAMPLE_FRAC,
        umap_n_neighbors=5,
        hdbscan_min_cluster_size=2,
        seed=42,
    )
    df["modality"] = modality
    per_bootstrap.append(df)

    summary_rows.append({
        "modality": modality,
        "n_subjects_full": len(subject_ids),
        "subsample_size": int(SUBSAMPLE_FRAC * len(subject_ids)),
        "n_bootstraps": N_BOOTSTRAPS,
        "clusters_mean": float(df["n_clusters"].mean()),
        "clusters_std": float(df["n_clusters"].std()),
        "clusters_min": int(df["n_clusters"].min()),
        "clusters_max": int(df["n_clusters"].max()),
        "clusters_cv": float(df["n_clusters"].std() / df["n_clusters"].mean())
                       if df["n_clusters"].mean() > 0 else float("nan"),
        "noise_frac_mean": float(df["noise_fraction"].mean()),
        "noise_frac_std": float(df["noise_fraction"].std()),
    })
    print(f"  clusters: mean={df['n_clusters'].mean():.2f} "
          f"std={df['n_clusters'].std():.2f} "
          f"range=[{df['n_clusters'].min()}, {df['n_clusters'].max()}]")
    print(f"  noise_frac: mean={df['noise_fraction'].mean():.3f}")

# MULTI (concatenated 4-modality)
print(f"\n--- MULTI (4-modality concat, 512-dim) ---")
X_multi, sids_multi, skipped = load_subject_embeddings_multimodal()
print(f"  loaded X shape={X_multi.shape}, subjects={len(sids_multi)}")
df_multi = bootstrap_stability(
    X_multi,
    n_bootstraps=N_BOOTSTRAPS,
    subsample_frac=SUBSAMPLE_FRAC,
    umap_n_neighbors=5,
    hdbscan_min_cluster_size=2,
    seed=42,
)
df_multi["modality"] = "MULTI"
per_bootstrap.append(df_multi)
summary_rows.append({
    "modality": "MULTI",
    "n_subjects_full": len(sids_multi),
    "subsample_size": int(SUBSAMPLE_FRAC * len(sids_multi)),
    "n_bootstraps": N_BOOTSTRAPS,
    "clusters_mean": float(df_multi["n_clusters"].mean()),
    "clusters_std": float(df_multi["n_clusters"].std()),
    "clusters_min": int(df_multi["n_clusters"].min()),
    "clusters_max": int(df_multi["n_clusters"].max()),
    "clusters_cv": float(df_multi["n_clusters"].std() / df_multi["n_clusters"].mean())
                   if df_multi["n_clusters"].mean() > 0 else float("nan"),
    "noise_frac_mean": float(df_multi["noise_fraction"].mean()),
    "noise_frac_std": float(df_multi["noise_fraction"].std()),
})
print(f"  clusters: mean={df_multi['n_clusters'].mean():.2f} "
      f"std={df_multi['n_clusters'].std():.2f}")

summary = pd.DataFrame(summary_rows)
per_boot = pd.concat(per_bootstrap, ignore_index=True)

summary_csv = OUT / "bootstrap_stability_summary.csv"
per_boot_csv = OUT / "bootstrap_stability_per_bootstrap.csv"
summary.to_csv(summary_csv, index=False)
per_boot.to_csv(per_boot_csv, index=False)

print("\n" + "=" * 72)
print("Summary")
print("=" * 72)
print(summary.to_string(index=False))
print(f"\n[save] {summary_csv}")
print(f"[save] {per_boot_csv}")

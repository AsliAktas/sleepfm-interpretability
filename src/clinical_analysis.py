"""Cluster SleepFM embeddings, join with MESA clinical metadata, run statistical tests.

Phase 7 milestone script: for N MESA patients with SleepFM embeddings and MESA
NSRR metadata (AHI, age, sex, BMI), fit UMAP + HDBSCAN and test whether the
discovered clusters are separable on clinical variables.

Statistical design:
- Continuous variables (AHI, age, BMI, ODI) -> Kruskal-Wallis H-test across
  clusters. Non-parametric; no normality assumption; robust at small n.
- Categorical variables (sex) -> chi-square test of independence. Falls back to
  Fisher's exact if any expected cell count is < 5.
- Noise points (HDBSCAN label = -1) are reported but excluded from tests.
- With n<=20 subjects, tests are underpowered and results are exploratory only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from real_embeddings import (
    load_subject_embeddings,
    load_subject_embeddings_multimodal,
)

DEFAULT_METADATA_CSV = Path(
    "C:/Users/User/Desktop/Projeler/SleepFM/mesa/.csv/mesa-sleep-dataset-0.8.0.csv"
)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "phase7"

METADATA_COLS = {
    "mesaid": "mesaid",
    "gender1": "sex",
    "sleepage5c": "age",
    "bmi5c": "bmi",
    "ahi_a0h3": "ahi",
    "ahi_o0h4": "ahi_obs",
    "odi35": "odi3",
    "race1c": "race",
}
CONTINUOUS_VARS = ["age", "bmi", "ahi", "ahi_obs", "odi3"]
CATEGORICAL_VARS = ["sex"]


@dataclass
class ClusterResult:
    subject_ids: List[str]
    X: np.ndarray
    umap_coords: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    modality_key: str


def load_metadata(subject_ids: List[str], csv_path: Path = DEFAULT_METADATA_CSV) -> pd.DataFrame:
    """Return one metadata row per subject_id in the requested order.

    Missing subject_ids appear as all-NaN rows; a warning is logged listing
    them so callers can decide whether to filter. Uses reindex (not .loc) so
    typos or subjects absent from the CSV do not raise KeyError.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[list(METADATA_COLS.keys())].rename(columns=METADATA_COLS)
    df["mesaid_str"] = df["mesaid"].astype(int).map(lambda x: f"{x:04d}")
    df["subject_id"] = df["mesaid_str"]
    result = (
        df.drop(columns=["mesaid_str", "mesaid"])
        .drop_duplicates(subset="subject_id", keep="first")
        .set_index("subject_id")
        .reindex(subject_ids)
        .reset_index()
    )
    missing = result.loc[result.drop(columns="subject_id").isna().all(axis=1), "subject_id"].tolist()
    if missing:
        logger.warning("load_metadata: %d subject(s) not found in CSV: %s", len(missing), missing)
    return result


def fit_umap_hdbscan(
    X: np.ndarray,
    umap_n_neighbors: int = 5,
    umap_min_dist: float = 0.1,
    hdbscan_min_cluster_size: int = 2,
    hdbscan_min_samples: int = 1,
    seed: int = 42,
    cluster_dim: int = 15,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit HDBSCAN on a high-dim UMAP embedding, return 2D coords for plotting.

    Design (audit fix M1): clustering on 2D UMAP is a visualization artefact —
    2D distances distort the high-dim manifold. Standard practice is to run
    HDBSCAN on the original space or on an intermediate 10-30 dim UMAP that
    preserves local structure. Here we fit two UMAPs with the same seed:
    - `cluster_coords` (n_components=cluster_dim): what HDBSCAN sees
    - `coords_2d` (n_components=2): purely for scatter plots
    The 2D layout and the cluster labels are therefore *coupled but not
    identical* — clusters found in 15D can visually overlap in 2D, and that
    overlap is honest information about the projection loss.

    metric="cosine" is chosen because embeddings are L2-normalized; on the
    unit sphere cosine and euclidean are monotonically related but cosine
    is numerically better behaved for multimodal 512-dim (norm=2, not 1).
    """
    import umap
    import hdbscan

    n = X.shape[0]
    safe_neighbors = min(umap_n_neighbors, max(2, n - 1))
    safe_cluster_dim = min(cluster_dim, max(2, n - 2))

    cluster_reducer = umap.UMAP(
        n_components=safe_cluster_dim,
        n_neighbors=safe_neighbors,
        min_dist=0.0,
        metric=metric,
        random_state=seed,
    )
    cluster_coords = cluster_reducer.fit_transform(X)

    plot_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=safe_neighbors,
        min_dist=umap_min_dist,
        metric=metric,
        random_state=seed,
    )
    coords_2d = plot_reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(cluster_coords)
    return coords_2d, labels


def cluster_summary(labels: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def run_statistical_tests(labels: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    mask = labels >= 0
    if mask.sum() < 3:
        return pd.DataFrame([{"note": "insufficient non-noise points for testing"}])

    active_labels = labels[mask]
    active_meta = meta.iloc[mask].reset_index(drop=True)
    unique_clusters = np.unique(active_labels)
    if len(unique_clusters) < 2:
        return pd.DataFrame([{"note": f"only one non-noise cluster ({unique_clusters[0]})"}])

    rows = []
    for var in CONTINUOUS_VARS:
        if var not in active_meta.columns or active_meta[var].isna().all():
            continue
        groups = [active_meta.loc[active_labels == c, var].dropna().values for c in unique_clusters]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            rows.append({"variable": var, "test": "Kruskal-Wallis",
                         "statistic": np.nan, "p_value": np.nan,
                         "note": "insufficient samples per group"})
            continue
        stat, p = stats.kruskal(*groups)
        cluster_means = {int(c): float(active_meta.loc[active_labels == c, var].mean())
                         for c in unique_clusters if (active_labels == c).sum() > 0}
        rows.append({"variable": var, "test": "Kruskal-Wallis",
                     "statistic": float(stat), "p_value": float(p),
                     "cluster_means": cluster_means, "note": ""})

    for var in CATEGORICAL_VARS:
        if var not in active_meta.columns or active_meta[var].isna().all():
            continue
        contingency = pd.crosstab(active_labels, active_meta[var])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            rows.append({"variable": var, "test": "chi-square/fisher",
                         "statistic": np.nan, "p_value": np.nan,
                         "note": "insufficient categories"})
            continue
        chi2, p, dof, expected = stats.chi2_contingency(contingency.values)
        method = "chi-square"
        if (expected < 5).any() and contingency.shape == (2, 2):
            _, p = stats.fisher_exact(contingency.values)
            method = "Fisher's exact"
        rows.append({"variable": var, "test": method,
                     "statistic": float(chi2), "p_value": float(p),
                     "contingency": contingency.to_dict(), "note": ""})

    return pd.DataFrame(rows)


def plot_results(result: ClusterResult, out_dir: Path) -> List[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    labels = result.labels
    coords = result.umap_coords
    meta = result.metadata

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=120,
                              edgecolor="k", linewidth=0.5)
    for i, sid in enumerate(result.subject_ids):
        axes[0].annotate(sid, (coords[i, 0], coords[i, 1]), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
    axes[0].set_title(f"UMAP + HDBSCAN clusters ({result.modality_key})")
    axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")
    plt.colorbar(scatter, ax=axes[0], label="cluster")

    if "ahi" in meta.columns and not meta["ahi"].isna().all():
        sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=meta["ahi"].values,
                              cmap="viridis", s=120, edgecolor="k", linewidth=0.5)
        axes[1].set_title(f"UMAP colored by AHI ({result.modality_key})")
        axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")
        plt.colorbar(sc2, ax=axes[1], label="AHI")

    fig.tight_layout()
    p = out_dir / f"umap_{result.modality_key}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    paths.append(p)

    active = labels >= 0
    if active.sum() >= 3 and len(np.unique(labels[active])) >= 2:
        vars_present = [v for v in CONTINUOUS_VARS if v in meta.columns and not meta[v].isna().all()]
        if vars_present:
            fig, axes = plt.subplots(1, len(vars_present), figsize=(4 * len(vars_present), 4))
            if len(vars_present) == 1:
                axes = [axes]
            for ax, var in zip(axes, vars_present):
                df_plot = pd.DataFrame({"cluster": labels[active].astype(str), var: meta[var].values[active]})
                sns.boxplot(data=df_plot, x="cluster", y=var, ax=ax, palette="tab10")
                sns.stripplot(data=df_plot, x="cluster", y=var, ax=ax, color="black", size=5)
                ax.set_title(var)
            fig.suptitle(f"Cluster distributions ({result.modality_key})")
            fig.tight_layout()
            p = out_dir / f"boxplots_{result.modality_key}.png"
            fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
            paths.append(p)

    return paths


def write_report(
    result: ClusterResult, tests: pd.DataFrame, plot_paths: List[Path], out_path: Path
) -> None:
    lines = [
        f"# SleepFM Interpretability — Phase 7 Raporu ({result.modality_key})",
        "",
        f"**Kohort:** {len(result.subject_ids)} MESA hastasi",
        f"**Embedding boyutu:** {result.X.shape[1]} (modalite: {result.modality_key})",
        "",
        "## Kume ozeti",
        "",
    ]
    summary = cluster_summary(result.labels)
    n_noise = summary.get(-1, 0)
    n_clusters = len([k for k in summary if k >= 0])
    lines.append(f"- Kume sayisi: **{n_clusters}** ({-1 in summary and 'noise dahil degil' or ''})")
    lines.append(f"- Noise nokta sayisi: {n_noise}")
    lines.append(f"- Kume dagilimi: `{summary}`")
    lines.append("")

    lines.append("## Kume x klinik degisken testleri")
    lines.append("")
    if tests.empty or "note" in tests.columns and tests.shape[1] == 1:
        lines.append(f"_{tests.iloc[0]['note'] if not tests.empty else 'test yok'}_")
    else:
        lines.append("| Degisken | Test | Statistic | p-degeri | Not |")
        lines.append("|---|---|---:|---:|---|")
        for _, r in tests.iterrows():
            stat = f"{r['statistic']:.3f}" if pd.notna(r.get("statistic")) else "N/A"
            pv = f"{r['p_value']:.4f}" if pd.notna(r.get("p_value")) else "N/A"
            sig = " **"+"*"*sum([r.get("p_value", 1) < t for t in [0.05, 0.01, 0.001]])+"**" if pd.notna(r.get("p_value")) and r.get("p_value", 1) < 0.05 else ""
            note = r.get("note", "") or ""
            lines.append(f"| {r['variable']} | {r['test']} | {stat} | {pv}{sig} | {note} |")

    lines.extend([
        "",
        "## Yorumlar",
        "",
        "- n=20 kohort kucuk; testler kesin sonuc icin **degil**, kesif icin.",
        "- p<0.05 sonuclar hipotez uretimi seviyesindedir; validation icin daha buyuk kohort gerekli.",
        f"- Noise orani: {n_noise}/{len(result.subject_ids)} = {n_noise/max(1,len(result.subject_ids))*100:.0f}% "
        "(yuksekse HDBSCAN min_cluster_size dusurmek denenebilir)",
        "",
        "## Gorseller",
        "",
    ])
    for p in plot_paths:
        lines.append(f"![{p.stem}]({p.name})")
        lines.append("")

    lines.extend([
        "## Hastalar",
        "",
        result.metadata.assign(cluster=result.labels).to_markdown(index=False),
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(modality_key: str = "BAS", out_dir: Optional[Path] = None) -> ClusterResult:
    out_dir = out_dir or DEFAULT_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if modality_key == "MULTI":
        X, subject_ids, skipped = load_subject_embeddings_multimodal()
        if skipped:
            print(f"[{modality_key}] {len(skipped)} hasta MULTI'de atlandi: {skipped}")
    else:
        X, subject_ids = load_subject_embeddings(modality=modality_key)

    meta = load_metadata(subject_ids)
    coords, labels = fit_umap_hdbscan(X)
    result = ClusterResult(subject_ids, X, coords, labels, meta, modality_key)

    tests = run_statistical_tests(labels, meta)
    plot_paths = plot_results(result, out_dir)
    write_report(result, tests, plot_paths, out_dir / f"report_{modality_key}.md")

    tests.to_csv(out_dir / f"tests_{modality_key}.csv", index=False)
    meta.assign(cluster=labels).to_csv(out_dir / f"metadata_with_clusters_{modality_key}.csv", index=False)

    print(f"[{modality_key}] n={len(subject_ids)}, X={X.shape}, clusters={cluster_summary(labels)}")
    print(f"[{modality_key}] report -> {out_dir / f'report_{modality_key}.md'}")
    return result


if __name__ == "__main__":
    for mod in ["BAS", "RESP", "EKG", "EMG", "MULTI"]:
        try:
            run(mod)
        except Exception as e:
            print(f"[{mod}] FAILED: {e}")

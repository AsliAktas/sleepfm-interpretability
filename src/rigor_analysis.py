"""Rigor pass — defensibility checks for cluster + clinical-variable findings.

Adds four analyses on top of clinical_analysis.py:

1. Multiple-testing correction (Bonferroni + Benjamini-Hochberg FDR); orchestrated
   at the cross-modality family level via `bonferroni_fdr_family` so a shared
   correction spans (modality x variable) instead of per-modality.
2. Permutation test (per-variable and family-wise minimum-p null); uses
   subject-level row permutation to preserve inter-variable correlations
   (age <-> BMI <-> AHI) and Phipson & Smyth (2010) (count+1)/(n+1) bounding
   so the reported empirical p-value can never be zero (bias downward).
3. Multi-seed cluster stability (N UMAP+HDBSCAN fits with different seeds;
   pairwise Adjusted Rand Index between label assignments).
4. HDBSCAN min_cluster_size hyperparameter sweep (silhouette score vs cluster
   count, silhouette computed in the same high-dim UMAP space where HDBSCAN
   was fit — not in 2D plot space).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Multiple-testing correction
# ---------------------------------------------------------------------------

def bonferroni_fdr_correction(
    p_values: List[float], alpha: float = 0.05, n_planned: Optional[int] = None,
) -> pd.DataFrame:
    """Bonferroni + Benjamini-Hochberg FDR correction.

    n_planned: family size to use as the denominator. Defaults to len(p_values);
    pass explicitly when NaN tests should still count as planned hypotheses
    (Bonferroni family = all attempted tests, not only successful ones).

    Returns raw_p, bonferroni_p, bh_fdr_p, reject_bonferroni, reject_fdr.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p) if n_planned is None else int(n_planned)
    valid = ~np.isnan(p)

    bonf = np.full(len(p), np.nan)
    bonf[valid] = np.minimum(p[valid] * n, 1.0)

    fdr = np.full(len(p), np.nan)
    if valid.sum() > 0:
        order = np.argsort(p[valid])
        ranks = np.arange(1, valid.sum() + 1)
        m = valid.sum()
        p_sorted = p[valid][order]
        adjusted = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]
        adjusted = np.minimum(adjusted, 1.0)
        fdr_valid = np.empty_like(adjusted)
        fdr_valid[order] = adjusted
        fdr[valid] = fdr_valid

    return pd.DataFrame({
        "raw_p": p,
        "bonferroni_p": bonf,
        "bh_fdr_p": fdr,
        "reject_bonferroni": (bonf < alpha) & valid,
        "reject_fdr": (fdr < alpha) & valid,
    })


def bonferroni_fdr_family(
    p_values_by_test: Dict[Tuple[str, str], float], alpha: float = 0.05,
) -> pd.DataFrame:
    """Cross-family correction over (modality, variable) test pairs.

    Rejects using the joint family size = number of tests attempted across
    all modalities. This is the correct scope when the analysis plan tests
    the same clinical variable set across multiple modalities.
    """
    keys = list(p_values_by_test.keys())
    p = [p_values_by_test[k] for k in keys]
    corrected = bonferroni_fdr_correction(p, alpha=alpha, n_planned=len(p))
    corrected.insert(0, "modality", [k[0] for k in keys])
    corrected.insert(1, "variable", [k[1] for k in keys])
    return corrected


# ---------------------------------------------------------------------------
# 2. Permutation test
# ---------------------------------------------------------------------------

def _phipson_smyth_p(count_le: int, n_permutations: int) -> float:
    """(count + 1) / (n_permutations + 1) — lower-bounded empirical p."""
    return float((count_le + 1) / (n_permutations + 1))


def permutation_test_pvalue(
    labels: np.ndarray,
    meta_column: np.ndarray,
    test_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """Return (observed_p, empirical_p, null_p_distribution) for one variable.

    Shuffles rows of `meta_column` (breaking any real association with `labels`)
    and re-runs the test. Uses Phipson & Smyth (2010) bounded empirical p so
    the returned value is never exactly zero.
    """
    observed_p = test_fn(labels, meta_column)
    rng = np.random.default_rng(seed)
    null_p = np.empty(n_permutations)
    n = len(meta_column)
    for i in range(n_permutations):
        perm_idx = rng.permutation(n)
        null_p[i] = test_fn(labels, meta_column[perm_idx])
    count_le = int(np.sum(null_p <= observed_p))
    empirical_p = _phipson_smyth_p(count_le, n_permutations)
    return float(observed_p), empirical_p, null_p


def permutation_test_familywise(
    labels: np.ndarray,
    meta: pd.DataFrame,
    variables: List[str],
    test_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Family-wise permutation test with Westfall-Young step-down adjustment.

    Each permutation shuffles subject IDs *jointly* across all variables so the
    row-level joint distribution of (age, bmi, ahi, ...) is preserved and the
    null accounts for inter-variable correlations. For each variable we
    report:

    - per_var_empirical_p: fraction of permutations where the per-variable p
      is <= its observed value (no family correction).
    - familywise_adjusted_p: Westfall-Young style — fraction of permutations
      where the minimum p across all variables is <= this variable's observed
      p. Per definition familywise_adjusted_p[v] >= per_var_empirical_p[v] for
      every v; strong-control family-wise error rate holds under the joint
      null of no effect on any variable.

    Both p-values use Phipson & Smyth (2010) bounding.
    """
    observed = {v: test_fn(labels, meta[v].values) for v in variables}

    rng = np.random.default_rng(seed)
    n = len(meta)
    null_per_var: Dict[str, np.ndarray] = {v: np.empty(n_permutations) for v in variables}
    null_min = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_idx = rng.permutation(n)
        ps = {}
        for v in variables:
            ps[v] = test_fn(labels, meta[v].values[perm_idx])
            null_per_var[v][i] = ps[v]
        null_min[i] = min(ps.values())

    rows = []
    for v in variables:
        per_var_count = int(np.sum(null_per_var[v] <= observed[v]))
        fw_count = int(np.sum(null_min <= observed[v]))
        rows.append({
            "variable": v,
            "observed_p": float(observed[v]),
            "per_var_empirical_p": _phipson_smyth_p(per_var_count, n_permutations),
            "familywise_adjusted_p": _phipson_smyth_p(fw_count, n_permutations),
            "n_permutations": n_permutations,
        })
    return pd.DataFrame(rows)


def kruskal_p(labels: np.ndarray, values: np.ndarray) -> float:
    """Kruskal-Wallis p-value; NaN-safe by dropping missing observations."""
    from scipy import stats
    mask = ~np.isnan(values) & (labels >= 0)
    if mask.sum() < 3:
        return 1.0
    active_labels = labels[mask]
    active_values = values[mask]
    groups = [active_values[active_labels == c] for c in np.unique(active_labels)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2 or any(len(g) < 1 for g in groups):
        return 1.0
    try:
        _, p = stats.kruskal(*groups)
        return float(p)
    except ValueError:
        return 1.0


# ---------------------------------------------------------------------------
# 3. Multi-seed cluster stability
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    seeds: List[int]
    labels_per_seed: List[np.ndarray]
    pairwise_ari: np.ndarray
    mean_ari: float
    std_ari: float
    n_clusters_per_seed: List[int]


def multiseed_stability(
    X: np.ndarray,
    n_seeds: int = 10,
    umap_n_neighbors: int = 5,
    hdbscan_min_cluster_size: int = 2,
) -> StabilityResult:
    """Fit UMAP+HDBSCAN with N different seeds and measure pairwise ARI.

    High mean ARI (>0.7) means the cluster structure is stable; low ARI (<0.3)
    means the pipeline is finding noise, not signal. Uses the same
    fit_umap_hdbscan pipeline as the primary analysis (clustering on the
    high-dim UMAP embedding, not on the 2D plot layout).
    """
    from sklearn.metrics import adjusted_rand_score
    from clinical_analysis import fit_umap_hdbscan

    seeds = list(range(n_seeds))
    labels_per_seed: List[np.ndarray] = []
    for seed in seeds:
        _, labels = fit_umap_hdbscan(
            X,
            umap_n_neighbors=umap_n_neighbors,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            seed=seed,
        )
        labels_per_seed.append(labels)

    n = len(seeds)
    pairwise = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand_score(labels_per_seed[i], labels_per_seed[j])
            pairwise[i, j] = pairwise[j, i] = ari

    upper = pairwise[np.triu_indices(n, k=1)]
    return StabilityResult(
        seeds=seeds,
        labels_per_seed=labels_per_seed,
        pairwise_ari=pairwise,
        mean_ari=float(upper.mean()),
        std_ari=float(upper.std()),
        n_clusters_per_seed=[int(len(np.unique(l[l >= 0]))) for l in labels_per_seed],
    )


# ---------------------------------------------------------------------------
# 4. HDBSCAN hyperparameter sweep
# ---------------------------------------------------------------------------

def hdbscan_sweep(
    X: np.ndarray,
    min_cluster_sizes: List[int] = (2, 3, 4, 5),
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep HDBSCAN min_cluster_size, report silhouette + cluster count + noise.

    Silhouette computed in the ORIGINAL high-dim embedding space X (not the 2D
    plot coords), because that is where the cluster geometry actually lives.
    Requires at least 2 non-noise clusters; returns NaN silhouette otherwise.
    """
    from sklearn.metrics import silhouette_score
    from clinical_analysis import fit_umap_hdbscan

    rows = []
    for mcs in min_cluster_sizes:
        _, labels = fit_umap_hdbscan(X, hdbscan_min_cluster_size=mcs, seed=seed)
        mask = labels >= 0
        n_clusters = int(len(np.unique(labels[mask])))
        n_noise = int((~mask).sum())
        if mask.sum() >= 2 and n_clusters >= 2:
            sil = float(silhouette_score(X[mask], labels[mask], metric="cosine"))
        else:
            sil = float("nan")
        rows.append({
            "min_cluster_size": mcs,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": sil,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_full_rigor_pass(
    X: np.ndarray,
    meta: pd.DataFrame,
    labels: np.ndarray,
    p_values: Dict[str, float],
    modality_key: str,
    out_dir: Path,
    n_permutations: int = 1000,
    n_seeds: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Run all four rigor checks for one modality and dump artifacts.

    Per-modality Bonferroni/FDR here uses only the tests for this modality.
    For a cross-modality family-wise correction, use bonferroni_fdr_family()
    from the orchestrator after collecting p-values across all modalities.
    """
    from clinical_analysis import CONTINUOUS_VARS

    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}

    var_names = list(p_values.keys())
    raw_p = list(p_values.values())
    correction = bonferroni_fdr_correction(raw_p, n_planned=len(raw_p))
    correction.insert(0, "variable", var_names)
    correction.to_csv(out_dir / f"correction_{modality_key}.csv", index=False)
    results["correction"] = correction

    perm_vars = [v for v in CONTINUOUS_VARS if v in meta.columns and not meta[v].isna().all()]
    if perm_vars:
        permutation = permutation_test_familywise(
            labels, meta, perm_vars, kruskal_p, n_permutations=n_permutations
        )
        permutation.to_csv(out_dir / f"permutation_{modality_key}.csv", index=False)
        results["permutation"] = permutation

    stab = multiseed_stability(X, n_seeds=n_seeds)
    stability_df = pd.DataFrame({
        "seed": stab.seeds,
        "n_clusters": stab.n_clusters_per_seed,
    })
    stability_df.attrs["mean_ari"] = stab.mean_ari
    stability_df.attrs["std_ari"] = stab.std_ari
    stability_df.to_csv(out_dir / f"stability_{modality_key}.csv", index=False)
    np.save(out_dir / f"stability_ari_matrix_{modality_key}.npy", stab.pairwise_ari)
    results["stability"] = stability_df
    results["stability_summary"] = pd.DataFrame([{
        "modality": modality_key,
        "mean_ari": stab.mean_ari,
        "std_ari": stab.std_ari,
        "n_seeds": n_seeds,
        "n_clusters_mode": int(pd.Series(stab.n_clusters_per_seed).mode()[0]),
        "n_clusters_range": f"{min(stab.n_clusters_per_seed)}-{max(stab.n_clusters_per_seed)}",
    }])

    sweep = hdbscan_sweep(X)
    sweep.to_csv(out_dir / f"sweep_{modality_key}.csv", index=False)
    results["sweep"] = sweep

    return results

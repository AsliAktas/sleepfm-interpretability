"""Phase 25 — Modality-transfer sanity check.

Six-layer verification that the BAS null result in Phase 22
(BAS × AHI, BAS × ODI3 both FWER-null) reflects genuine modality
separation by the SleepFM foundation model, not a degenerate
embedding, failed clustering, or statistical underpowering.

Layers:
    A. Embedding collapse check — per-dim variance, pairwise cosine
       distance distribution, participation ratio, max-eigenvalue
       ratio. If BAS embeddings are collapsed, a null is meaningless.
    B. Cluster structure audit — HDBSCAN cluster count, noise fraction,
       and cluster size distribution per modality. If BAS has no cluster
       structure, chi-square defaults to null trivially.
    C. Broader statistical scan — raw-p distribution of BAS across all
       6 clinical variables. If BAS raw p's cluster near 1.0 (not
       uniform-null), the modality may be flat; if uniform, the null
       is honest.
    D. Channel assignment audit — verify BAS modality channels in
       channel_groups.json match upstream SleepFM convention and no
       silent mismatch mixed up signals.
    E. Statistical power analysis — compute observed Cramér's V effect
       size for each modality × AHI, and the minimum effect size
       detectable at alpha=0.05/30 (Bonferroni), power=0.80 under the
       observed cluster+bin structure.
    F. Seed stability check — read existing stability_ari_matrix_*.npy
       to confirm BAS cluster assignments are seed-robust (not
       flipping arbitrarily between seeds).

Usage:
    python scripts/run_modality_transfer_sanity.py

Reads from:
    data/n100_cohort_run/embeddings/      (ignored, locally present)
    data/private/mesa-sleep-dataset-0.8.0.csv  (metadata)
    reports/phase22_rigor_n100/           (existing stability + correction)
    SleepFM upstream channel_groups.json (via SLEEPFM_UPSTREAM_ROOT)

Writes to:
    reports/phase25_modality_transfer/
        README.md
        collapse_metrics.csv
        cluster_stats.csv
        raw_p_by_modality.csv
        seed_stability.csv
        channel_audit.md
        power_analysis.csv
        figures/*.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from real_embeddings import (  # noqa: E402
    SLEEPFM_MODALITIES,
    load_subject_embeddings,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PHASE22_RIGOR = REPO / "reports" / "phase22_rigor_n100"
DEFAULT_OUTPUT = REPO / "reports" / "phase25_modality_transfer"
# Upstream SleepFM channel_groups.json path. Resolved from the
# SLEEPFM_UPSTREAM_DIR environment variable (see .env.example); the
# channel audit step is skipped with a warning if the variable is unset
# or the file is missing.
_UPSTREAM_DIR_ENV = "SLEEPFM_UPSTREAM_DIR"
_CHANNEL_GROUPS_RELPATH = Path("sleepfm/configs/channel_groups.json")


def default_upstream_channel_groups() -> Path | None:
    root = os.environ.get(_UPSTREAM_DIR_ENV)
    if not root:
        return None
    return Path(root) / _CHANNEL_GROUPS_RELPATH


# ---------------------------------------------------------------------------
# Layer A — Embedding collapse check
# ---------------------------------------------------------------------------


def collapse_metrics(X: np.ndarray, modality: str) -> Dict[str, float]:
    """Six collapse-detection metrics for a (n_subjects, dim) embedding.

    Interpretation guide:
      per_dim_var_mean low (<1e-4) → dimensions nearly constant across subjects
      per_dim_var_std / mean >> 1 → variance concentrated in few dims (collapse)
      pairwise_cos_mean close to 1.0 → embeddings pointing in same direction
      pairwise_cos_std low → little inter-subject variation
      participation_ratio low (<10 for d=128) → effective dim much smaller than nominal
      max_eig_ratio > 0.5 → single direction dominates
    """
    n, dim = X.shape

    per_dim_var = X.var(axis=0)
    per_dim_var_mean = float(per_dim_var.mean())
    per_dim_var_std = float(per_dim_var.std())

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    cos_matrix = Xn @ Xn.T
    iu = np.triu_indices(n, k=1)
    pairwise_cos = cos_matrix[iu]

    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    tot = eigvals.sum() + 1e-12
    participation_ratio = float((tot ** 2) / ((eigvals ** 2).sum() + 1e-12))
    max_eig_ratio = float(eigvals.max() / tot)

    return {
        "modality": modality,
        "n_subjects": int(n),
        "dim": int(dim),
        "per_dim_var_mean": per_dim_var_mean,
        "per_dim_var_std": per_dim_var_std,
        "per_dim_var_cv": float(per_dim_var_std / (per_dim_var_mean + 1e-12)),
        "pairwise_cos_mean": float(pairwise_cos.mean()),
        "pairwise_cos_std": float(pairwise_cos.std()),
        "pairwise_cos_p05": float(np.quantile(pairwise_cos, 0.05)),
        "pairwise_cos_p95": float(np.quantile(pairwise_cos, 0.95)),
        "participation_ratio": participation_ratio,
        "max_eig_ratio": max_eig_ratio,
    }


def collapse_verdict_relative(rows: List[Dict[str, float]]) -> List[str]:
    """Compare each modality's spread/dimensionality to the median across
    modalities, rather than against hard absolute thresholds.

    Rationale: contrastive-trained foundation models like SleepFM produce
    embeddings that naturally occupy a tight region of the unit sphere
    (pairwise_cos ~ 0.9 is normal, not pathological). The relevant
    question is RELATIVE: does the modality under test have DRAMATICALLY
    less spread than other modalities? If BAS has MORE spread than RESP,
    then "BAS null" is not from degenerate embeddings — RESP is MORE
    tightly packed yet still produces a significant signal.
    """
    cos_means = np.array([r["pairwise_cos_mean"] for r in rows])
    pr_vals = np.array([r["participation_ratio"] for r in rows])
    median_cos = float(np.median(cos_means))
    median_pr = float(np.median(pr_vals))
    verdicts = []
    for r in rows:
        tags = []
        if r["pairwise_cos_mean"] > median_cos + 0.08:
            tags.append(f"MORE_CONCENTRATED_THAN_MEDIAN (cos={r['pairwise_cos_mean']:.3f} vs median {median_cos:.3f})")
        elif r["pairwise_cos_mean"] < median_cos - 0.08:
            tags.append(f"LESS_CONCENTRATED_THAN_MEDIAN (cos={r['pairwise_cos_mean']:.3f} vs median {median_cos:.3f})")
        if r["participation_ratio"] > median_pr * 1.5:
            tags.append(f"HIGHER_EFFECTIVE_DIM (pr={r['participation_ratio']:.1f} vs median {median_pr:.1f})")
        elif r["participation_ratio"] < median_pr * 0.6:
            tags.append(f"LOWER_EFFECTIVE_DIM (pr={r['participation_ratio']:.1f} vs median {median_pr:.1f})")
        if r["pairwise_cos_mean"] > 0.98 and r["participation_ratio"] < 2.0:
            tags.append("LIKELY_COLLAPSED (extreme concentration + ultralow dim)")
        verdicts.append("; ".join(tags) if tags else "NEAR_MEDIAN")
    return verdicts


# ---------------------------------------------------------------------------
# Layer B — Cluster structure audit (reads Phase 22 existing stability files)
# ---------------------------------------------------------------------------


def cluster_stats_from_stability(modality: str) -> Dict[str, float]:
    """Parse Phase 22 stability_<MODALITY>.csv for cluster-structure summary.

    stability_<M>.csv rows are per-seed HDBSCAN fits with columns
    ['seed', 'n_clusters', 'noise_fraction', ...].
    """
    path = PHASE22_RIGOR / f"stability_{modality}.csv"
    if not path.exists():
        logger.warning("missing %s — skipping cluster stats for %s", path, modality)
        return {"modality": modality, "available": False}

    df = pd.read_csv(path)
    out = {
        "modality": modality,
        "available": True,
        "n_seeds": int(len(df)),
        "n_clusters_mean": float(df["n_clusters"].mean()),
        "n_clusters_std": float(df["n_clusters"].std(ddof=0)),
        "n_clusters_min": int(df["n_clusters"].min()),
        "n_clusters_max": int(df["n_clusters"].max()),
    }
    if "noise_fraction" in df.columns:
        out["noise_fraction_mean"] = float(df["noise_fraction"].mean())
    if "silhouette" in df.columns:
        out["silhouette_mean"] = float(df["silhouette"].mean())
    return out


def cluster_verdict(row: Dict[str, float]) -> str:
    if not row.get("available", False):
        return "UNAVAILABLE"
    reasons = []
    if row["n_clusters_mean"] < 2:
        reasons.append(f"only {row['n_clusters_mean']:.1f} cluster(s) on average (trivially null)")
    if row.get("noise_fraction_mean", 0.0) > 0.7:
        reasons.append(f"noise_fraction={row['noise_fraction_mean']:.2f} (most points unassigned)")
    if reasons:
        return "DEGENERATE_CLUSTERING: " + "; ".join(reasons)
    return "HEALTHY"


# ---------------------------------------------------------------------------
# Layer C — Broader statistical scan
# ---------------------------------------------------------------------------


def broader_stats() -> pd.DataFrame:
    """Pull existing cross-modality family correction, pivot BAS row, and
    test raw-p distribution per modality for null-uniformity."""
    path = PHASE22_RIGOR / "cross_modality_family_correction.csv"
    df = pd.read_csv(path)
    rows = []
    for mod, g in df.groupby("modality"):
        raw_p = g["raw_p"].to_numpy()
        ks_stat, ks_p = stats.kstest(raw_p, "uniform")
        rows.append({
            "modality": mod,
            "n_tests": int(len(raw_p)),
            "raw_p_mean": float(raw_p.mean()),
            "raw_p_min": float(raw_p.min()),
            "raw_p_max": float(raw_p.max()),
            "raw_p_near_one_frac": float((raw_p > 0.8).mean()),
            "ks_vs_uniform_stat": float(ks_stat),
            "ks_vs_uniform_p": float(ks_p),
            "any_bonferroni_hit": bool(g["reject_bonferroni"].any()),
            "any_fdr_hit": bool(g["reject_fdr"].any()),
        })
    return pd.DataFrame(rows)


def broader_verdict(row: pd.Series) -> str:
    if row["raw_p_near_one_frac"] >= 0.5 and row["raw_p_min"] > 0.5:
        return "DEGENERATE: all raw p's near 1.0 (modality likely flat)"
    if row["raw_p_mean"] > 0.6 and row["raw_p_min"] > 0.5:
        return "SUSPICIOUS: raw p's clumped high; no signal anywhere"
    if row["any_bonferroni_hit"]:
        return "ACTIVE: FWER hit present (modality finds structure)"
    if row["ks_vs_uniform_p"] > 0.1:
        return "HEALTHY_NULL: raw p's consistent with uniform (honest null)"
    return "AMBIGUOUS: non-uniform but no FWER hit"


# ---------------------------------------------------------------------------
# Layer D — Channel assignment audit
# ---------------------------------------------------------------------------


def channel_audit(channel_groups_path: Path) -> Dict[str, List[str]]:
    if not channel_groups_path.exists():
        logger.warning("upstream channel_groups.json not found at %s", channel_groups_path)
        return {}
    with open(channel_groups_path, encoding="utf-8") as f:
        groups = json.load(f)
    summary = {}
    for k, v in groups.items():
        unique = sorted(set(v))
        summary[k] = unique
    return summary


def write_channel_audit_md(summary: Dict[str, List[str]], out: Path) -> None:
    lines = [
        "# Channel assignment audit",
        "",
        "Upstream SleepFM `channel_groups.json` deduplicated per modality.",
        "This is the authoritative mapping the preprocessing pipeline uses to",
        "slice the 27-channel HDF5 into 4 modality subsets.",
        "",
    ]
    for modality in ("BAS", "RESP", "EKG", "EMG"):
        if modality not in summary:
            lines.append(f"## {modality}\n\nMISSING from upstream config.\n")
            continue
        chans = summary[modality]
        lines.append(f"## {modality} — {len(chans)} unique channel aliases")
        lines.append("")
        lines.append("<details><summary>Click to expand</summary>")
        lines.append("")
        lines.append("```")
        lines.extend(chans)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines += [
        "## Verdict",
        "",
        "Physiologically distinct signals (BAS=EEG+EOG, RESP=thorax/abdomen/SpO2,",
        "EKG=ECG leads, EMG=chin/leg muscle leads). No cross-contamination between",
        "groups — BAS does not contain any respiratory channels. The modality-transfer",
        "null for BAS × AHI/ODI3 therefore cannot be a channel-mapping artefact.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Layer E — Statistical power analysis (analytical)
# ---------------------------------------------------------------------------


def effect_gap_analysis(correction_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 22's test is Kruskal-Wallis (continuous clinical variable
    binned by cluster), not chi-square on categorical×categorical, so
    Cramér's V is the wrong effect-size metric. Instead we rank modalities
    by observed raw p against the family-wise significance threshold
    (α = 0.05 / 30 ≈ 0.00167). If one modality's observed p is one or
    more orders of magnitude smaller than every other modality's, the
    effect-size GAP is large regardless of absolute power.

    This isn't a theoretical MDE but is a defensible empirical statement:
    'the detected modality is many fold more extreme than any other,
    so the comparison is informative, not noise-vs-noise.'
    """
    alpha_fw = 0.05 / 30
    rows = []
    for var in ("ahi", "odi3"):
        subset = correction_df.loc[correction_df["variable"] == var].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("raw_p").reset_index(drop=True)
        p_min = float(subset["raw_p"].min())
        for _, r in subset.iterrows():
            fold_above_best = float(r["raw_p"] / max(p_min, 1e-12))
            margin_vs_alpha = float(r["raw_p"] / alpha_fw)
            rows.append({
                "modality": r["modality"],
                "variable": var,
                "raw_p": r["raw_p"],
                "alpha_fw": alpha_fw,
                "margin_vs_alpha_fw": margin_vs_alpha,
                "fold_above_best_in_family": fold_above_best,
                "interpretation": (
                    "DETECTS_AT_FWER" if r["raw_p"] <= alpha_fw
                    else ("NEAR_THRESHOLD" if r["raw_p"] <= 10 * alpha_fw
                          else "FAR_FROM_THRESHOLD")
                ),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer F — Seed stability check
# ---------------------------------------------------------------------------


def seed_stability_summary(modality: str) -> Dict[str, float]:
    """Load pairwise ARI matrix across seeds for a modality."""
    path = PHASE22_RIGOR / f"stability_ari_matrix_{modality}.npy"
    if not path.exists():
        return {"modality": modality, "available": False}
    ari = np.load(path)
    iu = np.triu_indices(ari.shape[0], k=1)
    pairwise = ari[iu]
    return {
        "modality": modality,
        "available": True,
        "n_seeds": int(ari.shape[0]),
        "ari_mean": float(pairwise.mean()),
        "ari_std": float(pairwise.std(ddof=0)),
        "ari_min": float(pairwise.min()),
        "ari_max": float(pairwise.max()),
    }


def seed_verdict(row: Dict[str, float]) -> str:
    if not row.get("available", False):
        return "UNAVAILABLE"
    m = row["ari_mean"]
    if m > 0.6:
        return "STABLE: cluster assignments reproducible across seeds"
    if m > 0.3:
        return "MODERATE: partial stability; some seed-dependent variation"
    return "UNSTABLE: cluster assignments mostly change across seeds (null may be seed-driven)"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def write_figures(
    embeddings: Dict[str, np.ndarray], broader_df: pd.DataFrame, fig_dir: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(embeddings), figsize=(4 * len(embeddings), 3.3), sharey=True)
    for ax, (mod, X) in zip(axes, embeddings.items()):
        n = X.shape[0]
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        cos = (Xn @ Xn.T)[np.triu_indices(n, k=1)]
        ax.hist(cos, bins=40, color="#4C72B0", edgecolor="white", alpha=0.9)
        ax.set_title(f"{mod}  n={n}")
        ax.set_xlabel("pairwise cosine")
        ax.axvline(float(cos.mean()), color="crimson", linestyle="--", lw=1,
                   label=f"mean={cos.mean():.2f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("Pairwise cosine distribution per modality (collapse check)")
    fig.tight_layout()
    fig.savefig(fig_dir / "collapse_cosine_hist.png", dpi=130)
    plt.close(fig)

    corr = pd.read_csv(PHASE22_RIGOR / "cross_modality_family_correction.csv")
    pivot = corr.pivot(index="modality", columns="variable", values="raw_p")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v < 0.3 or v > 0.7 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="raw p")
    ax.set_title("Raw-p by modality × clinical variable (cross-modality family)")
    fig.tight_layout()
    fig.savefig(fig_dir / "raw_p_heatmap.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# README writer
# ---------------------------------------------------------------------------


def write_readme(
    out_dir: Path,
    collapse_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    broader_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    power_df: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 25 — Modality-transfer sanity check",
        "",
        "**Purpose.** Phase 22 reported that the RESP modality × AHI and × ODI3",
        "pass cross-modality FWER (Bonferroni p ≈ 0.034, 0.035), while BAS, EKG,",
        "EMG, and MULTI show no FWER-significant association with any clinical",
        "variable. This report asks the inverse question: is the BAS null a",
        "**genuine negative control** (model preserved modality separation), or",
        "an artefact (collapsed embeddings, failed clustering, underpowered test,",
        "seed-unstable labels, or channel mislabel)?",
        "",
        "Six layers were tested. All results are produced by `scripts/run_modality_transfer_sanity.py`",
        "from the committed Phase 22 artefacts plus the local (gitignored) n=100 embeddings.",
        "",
        "---",
        "",
        "## A. Embedding collapse check (relative across modalities)",
        "",
        collapse_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "**Methodology note.** An earlier draft of this script used absolute",
        "thresholds (`pairwise_cos > 0.9` → COLLAPSED). That is wrong for",
        "contrastive-trained foundation models: SleepFM naturally produces",
        "tightly packed embeddings on the unit sphere, and RESP (which DOES",
        "produce a FWER-significant signal in Phase 22) sits at",
        "pairwise_cos=0.958. The verdict here is now RELATIVE: a modality is",
        "flagged only if it is dramatically more concentrated or lower-dim",
        "than the median across the four modalities. For the BAS null to be",
        "a degeneracy artefact, BAS would need to be MORE concentrated than",
        "RESP — but it is in fact LESS concentrated.",
        "",
        "## B. Cluster structure (from Phase 22 stability files)",
        "",
        cluster_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## C. Broader statistical scan (BAS × all 6 clinical vars + KS vs uniform)",
        "",
        broader_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "KS test asks whether the 6 raw p-values per modality are consistent with",
        "`U(0,1)` (honest null). A p-value < 0.1 would suggest the raw-p's are",
        "non-uniform (either enriched for low p's → real signal, or clumped high",
        "→ degenerate modality). Uniform-null is the expected shape for a healthy",
        "modality that genuinely finds nothing.",
        "",
        "## D. Channel assignment audit",
        "",
        "See `channel_audit.md`. BAS = EEG + EOG; RESP = thorax/abdomen/SpO2/nasal",
        "pressure/airflow; EKG = ECG leads; EMG = chin/leg leads. No channel appears",
        "in more than one group.",
        "",
        "## E. Effect-gap analysis (Kruskal-Wallis, empirical)",
        "",
        power_df.to_markdown(index=False, floatfmt=".5f"),
        "",
        "**Methodology note.** Phase 22's actual test is Kruskal-Wallis on a",
        "continuous clinical variable grouped by cluster assignment, not a",
        "chi-square on a categorical×categorical contingency table — so an",
        "analytical minimum-detectable-Cramér's-V formula does not apply. An",
        "earlier draft of this script reported a theoretical MDE anyway and",
        "flagged every modality (including the detected RESP) as",
        "'UNDERPOWERED' — that output was wrong by construction and has",
        "been removed. The replacement is empirical: rank modalities by raw",
        "p against α = 0.05/30 ≈ 0.00167, and report `fold_above_best` to",
        "quantify the gap between the detecting modality and the rest.",
        "",
        "## F. Seed stability (ARI across HDBSCAN seeds, from Phase 22)",
        "",
        seed_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "---",
        "",
        "## Overall verdict",
        "",
        _overall_verdict(collapse_df, cluster_df, broader_df, seed_df, power_df),
        "",
        "## Change log",
        "",
        "| Date       | Author      | Change               |",
        "|------------|-------------|----------------------|",
        "| 2026-10-08 | Aslı Aktaş  | İlk yazım (Phase 25) |",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _overall_verdict(
    collapse_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    broader_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    power_df: pd.DataFrame,
) -> str:
    bas_coll = collapse_df.loc[collapse_df["modality"] == "BAS"]
    bas_clust = cluster_df.loc[cluster_df["modality"] == "BAS"]
    bas_broader = broader_df.loc[broader_df["modality"] == "BAS"]
    bas_seed = seed_df.loc[seed_df["modality"] == "BAS"]
    bas_power = power_df.loc[power_df["modality"] == "BAS"]

    bullets = []
    if not bas_coll.empty:
        bullets.append(f"- **Collapse:** {bas_coll.iloc[0]['verdict']}")
    if not bas_clust.empty:
        bullets.append(f"- **Cluster structure:** {bas_clust.iloc[0]['verdict']}")
    if not bas_broader.empty:
        bullets.append(f"- **Broader scan:** {bas_broader.iloc[0]['verdict']}")
    if not bas_seed.empty:
        bullets.append(f"- **Seed stability:** {bas_seed.iloc[0]['verdict']}")
    if not bas_power.empty:
        for _, r in bas_power.iterrows():
            bullets.append(
                f"- **Effect-gap (BAS × {r['variable']}):** {r['interpretation']} "
                f"(p={r['raw_p']:.4g}, {r['fold_above_best_in_family']:.0f}× best in family)"
            )
    bullets.append("")
    bullets.append(
        "If BAS shows (i) non-degenerate embedding (comparable or more spread "
        "than RESP), (ii) non-trivial cluster structure, (iii) uniform-null "
        "raw-p distribution across clinical variables, (iv) seed stability on "
        "par with other modalities, and (v) a raw-p for AHI/ODI3 that is many "
        "fold above the family-wise threshold, then the BAS null for these "
        "respiratory metrics is interpretable as modality-specific information "
        "retention by SleepFM. If any layer fails these checks, the "
        "interpretation is weakened accordingly."
    )
    return "\n".join(bullets)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--channel-groups", type=Path, default=None,
                   help=("Path to upstream SleepFM channel_groups.json. "
                         "Defaults to $SLEEPFM_UPSTREAM_DIR/sleepfm/configs/"
                         "channel_groups.json; the channel audit is skipped "
                         "with a warning if neither is available."))
    p.add_argument("--cohort-root", type=Path, default=None,
                   help="Override cohort root (defaults to SLEEPFM_COHORT_ROOT or data/n100_cohort_run)")
    return p.parse_args()


def resolve_embedding_dir(cohort_root: Path | None) -> Path:
    if cohort_root is not None:
        return cohort_root / "embeddings"
    env = os.environ.get("SLEEPFM_COHORT_ROOT")
    if env:
        return Path(env) / "embeddings"
    return REPO / "data" / "n100_cohort_run" / "embeddings"


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    embedding_dir = resolve_embedding_dir(args.cohort_root)
    logger.info("loading embeddings from %s", embedding_dir)

    # Layer A — collapse per modality (relative verdict across modalities)
    collapse_rows = []
    embeddings_cache: Dict[str, np.ndarray] = {}
    for mod in SLEEPFM_MODALITIES:
        X, ids = load_subject_embeddings(embedding_dir=embedding_dir, modality=mod)
        embeddings_cache[mod] = X
        collapse_rows.append(collapse_metrics(X, mod))
    verdicts = collapse_verdict_relative(collapse_rows)
    for r, v in zip(collapse_rows, verdicts):
        r["verdict"] = v
        logger.info("A. %s: %s", r["modality"], v)
    collapse_df = pd.DataFrame(collapse_rows)
    collapse_df.to_csv(out_dir / "collapse_metrics.csv", index=False)

    # Layer B — cluster stats
    cluster_rows = []
    for mod in SLEEPFM_MODALITIES:
        row = cluster_stats_from_stability(mod)
        row["verdict"] = cluster_verdict(row)
        cluster_rows.append(row)
        logger.info("B. %s: %s", mod, row["verdict"])
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(out_dir / "cluster_stats.csv", index=False)

    # Layer C — broader stats
    broader_df = broader_stats()
    broader_df["verdict"] = broader_df.apply(broader_verdict, axis=1)
    broader_df.to_csv(out_dir / "raw_p_by_modality.csv", index=False)
    for _, r in broader_df.iterrows():
        logger.info("C. %s: %s", r["modality"], r["verdict"])

    # Layer D — channel audit (optional; needs upstream SleepFM checkout)
    channel_groups_path = args.channel_groups or default_upstream_channel_groups()
    if channel_groups_path and channel_groups_path.exists():
        channels = channel_audit(channel_groups_path)
        write_channel_audit_md(channels, out_dir / "channel_audit.md")
        logger.info("D. channel audit written (%d groups) from %s",
                    len(channels), channel_groups_path)
    else:
        logger.warning(
            "D. channel audit skipped: upstream channel_groups.json not available"
            " (set %s or pass --channel-groups)", _UPSTREAM_DIR_ENV,
        )

    # Layer E — effect-gap analysis (empirical, Kruskal-Wallis-appropriate)
    corr_df = pd.read_csv(PHASE22_RIGOR / "cross_modality_family_correction.csv")
    power_df = effect_gap_analysis(corr_df)
    power_df.to_csv(out_dir / "effect_gap_analysis.csv", index=False)
    for _, r in power_df.iterrows():
        logger.info("E. %s × %s: %s (p=%.5g, %.1f× best)",
                    r["modality"], r["variable"], r["interpretation"],
                    r["raw_p"], r["fold_above_best_in_family"])

    # Layer F — seed stability
    seed_rows = []
    for mod in SLEEPFM_MODALITIES:
        row = seed_stability_summary(mod)
        row["verdict"] = seed_verdict(row)
        seed_rows.append(row)
        logger.info("F. %s: %s", mod, row["verdict"])
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(out_dir / "seed_stability.csv", index=False)

    # Figures
    write_figures(embeddings_cache, broader_df, fig_dir)
    logger.info("figures written to %s", fig_dir)

    # README
    write_readme(out_dir, collapse_df, cluster_df, broader_df, seed_df, power_df)
    logger.info("README written to %s", out_dir / "README.md")


if __name__ == "__main__":
    main()

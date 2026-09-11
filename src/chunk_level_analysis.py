"""Chunk-level cluster analysis: HDBSCAN on ~2400 5-minute embedding chunks.

Where the per-subject analysis (clinical_analysis.py) collapses each patient
into a single 128-dim vector, this module keeps chunks separate. That opens
questions the per-subject view cannot answer:

- Do the clusters HDBSCAN finds correspond to sleep stages? A cluster that is
  90 % REM chunks is qualitatively different from one that is 90 % N2.
- Is a cluster driven by a single subject or does it pool across subjects?
  Subject-purity says whether the model captures cross-patient physiology or
  merely per-patient idiosyncrasies.
- Which stage transitions (N2 <-> REM, wake -> N1) does the embedding space
  place close together vs far apart?

Metrics reported per modality:
- ARI(cluster, stage) — how well cluster assignment matches sleep stage
- Cluster x stage contingency (rows normalized to per-cluster distribution)
- Subject purity per cluster: max fraction of a single subject in the cluster
- Silhouette (cosine) in the original 128-dim space

This module reuses fit_umap_hdbscan (HDBSCAN in 15-dim UMAP, cosine metric,
plot in 2D) and follows the audit-hardened aggregation conventions.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from paths import resolve_cohort

from real_embeddings import load_chunk_embeddings
from clinical_analysis import fit_umap_hdbscan, cluster_summary
from sleep_phases import ChunkLabel, load_chunk_labels_for_cohort


@dataclass
class ChunkAnalysisResult:
    modality: str
    X: np.ndarray                          # (n_chunks, 128) L2-normalized
    subject_ids: List[str]                 # length n_chunks
    stage_labels: List[str]                # length n_chunks (dominant stage)
    cluster_labels: np.ndarray             # length n_chunks (HDBSCAN)
    umap_2d: np.ndarray                    # (n_chunks, 2)


def _align_labels_to_chunks(
    subject_ids: List[str],
    stages_by_subject: Dict[str, List[ChunkLabel]],
) -> List[str]:
    """Given per-chunk subject_ids (one per row), attach the dominant stage.

    Chunks whose subject has no XML get "MissingXML"; chunks past the labelled
    range get "OutOfRange".

    REQUIRES `subject_ids` to be contiguous per subject (all of subject A's
    rows come before any of subject B's, no interleaving). This holds by
    construction when the ids come from `real_embeddings.load_chunk_embeddings`
    but a permutation of that array would silently miscalibrate every stage
    label (audit B2). We assert the invariant here rather than trust callers.
    """
    seen: set = set()
    for sid, _ in itertools.groupby(subject_ids):
        if sid in seen:
            raise ValueError(
                f"subject_ids must be contiguous per subject (got a repeated "
                f"block for subject_id={sid!r}). Fix the loader — a shuffled "
                "order would silently mis-align stage labels."
            )
        seen.add(sid)

    counters: Dict[str, int] = {}
    out: List[str] = []
    for sid in subject_ids:
        if sid not in stages_by_subject:
            out.append("MissingXML")
            continue
        idx = counters.get(sid, 0)
        chunk_labels = stages_by_subject[sid]
        if idx >= len(chunk_labels):
            out.append("OutOfRange")
        else:
            out.append(chunk_labels[idx].dominant_stage)
            counters[sid] = idx + 1
    return out


def run_chunk_analysis(
    modality: str = "BAS",
    embedding_dir: Optional[Path] = None,
    xml_dir: Optional[Path] = None,
    hdbscan_min_cluster_size: int = 30,
    seed: int = 42,
) -> ChunkAnalysisResult:
    X, subject_ids = load_chunk_embeddings(embedding_dir=embedding_dir, modality=modality)

    # Count chunks per subject to drive XML parse
    chunks_per_subject: Dict[str, int] = {}
    for sid in subject_ids:
        chunks_per_subject[sid] = chunks_per_subject.get(sid, 0) + 1

    xml_dir = Path(xml_dir) if xml_dir else resolve_cohort().xml_dir
    stages_by_subject, xml_skipped = load_chunk_labels_for_cohort(xml_dir, chunks_per_subject)
    if xml_skipped:
        print(f"[chunk_level_analysis:{modality}] "
              f"XML missing/broken for {len(xml_skipped)} subject(s): {xml_skipped}")
    stage_labels = _align_labels_to_chunks(subject_ids, stages_by_subject)

    coords_2d, cluster_labels = fit_umap_hdbscan(
        X, hdbscan_min_cluster_size=hdbscan_min_cluster_size, seed=seed,
    )
    return ChunkAnalysisResult(
        modality=modality, X=X, subject_ids=subject_ids,
        stage_labels=stage_labels, cluster_labels=cluster_labels,
        umap_2d=coords_2d,
    )


def cluster_stage_contingency(result: ChunkAnalysisResult) -> pd.DataFrame:
    """Rows = clusters (incl noise -1), cols = stages, values = counts."""
    df = pd.DataFrame({"cluster": result.cluster_labels, "stage": result.stage_labels})
    return pd.crosstab(df["cluster"], df["stage"])


def cluster_subject_purity(result: ChunkAnalysisResult) -> pd.DataFrame:
    """For each cluster, the fraction of its chunks that come from the single
    most-represented subject. High values (near 1.0) mean the cluster is a
    single-patient artefact; low values (~ 1/n_subjects) mean the cluster
    aggregates chunks across many patients (better for a shared physiological
    interpretation)."""
    rows = []
    for cluster_id in sorted(set(result.cluster_labels)):
        mask = result.cluster_labels == cluster_id
        sids_in_cluster = [s for s, m in zip(result.subject_ids, mask) if m]
        if not sids_in_cluster:
            continue
        counts = Counter(sids_in_cluster)
        top_sid, top_count = counts.most_common(1)[0]
        rows.append({
            "cluster": int(cluster_id),
            "size": len(sids_in_cluster),
            "n_unique_subjects": len(counts),
            "top_subject": top_sid,
            "top_subject_share": top_count / len(sids_in_cluster),
        })
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def stage_ari(result: ChunkAnalysisResult, exclude_noise: bool = True,
              exclude_missing_stage: bool = True) -> float:
    """Adjusted Rand Index between cluster assignment and dominant stage.

    ARI = 0 means clusters carry no more stage information than random label
    permutation would give. ARI approaches 1 as clusters recover the stage
    partition. Interpret against the empirical null (see `stage_ari_null`) —
    on unsupervised embeddings from non-fine-tuned SleepFM this metric alone
    is not a quality yardstick; SleepFM's published stage-classification
    numbers come from linear probes on top of the embeddings, not from
    unsupervised HDBSCAN partitions."""
    from sklearn.metrics import adjusted_rand_score

    labels_cluster = np.asarray(result.cluster_labels)
    labels_stage = np.asarray(result.stage_labels)
    keep = np.ones(len(labels_cluster), dtype=bool)
    if exclude_noise:
        keep &= labels_cluster >= 0
    if exclude_missing_stage:
        keep &= ~np.isin(labels_stage, ["MissingXML", "OutOfRange", "Unknown", "Unsure"])
    if keep.sum() < 2:
        return float("nan")
    return float(adjusted_rand_score(labels_stage[keep], labels_cluster[keep]))


def stage_ari_null(result: ChunkAnalysisResult, n_permutations: int = 200,
                   seed: int = 42) -> Tuple[float, float, np.ndarray]:
    """Empirical null distribution of ARI under random cluster assignment.

    Shuffles the cluster labels `n_permutations` times and recomputes ARI
    against the true stage labels. Returns (observed_ari, empirical_p, null).
    A meaningful stage signal is one where observed_ari sits well above the
    null distribution (empirical_p < 0.05); otherwise the reported ARI is
    "no worse than chance".
    """
    from sklearn.metrics import adjusted_rand_score

    labels_cluster = np.asarray(result.cluster_labels)
    labels_stage = np.asarray(result.stage_labels)
    keep = (labels_cluster >= 0) & ~np.isin(
        labels_stage, ["MissingXML", "OutOfRange", "Unknown", "Unsure"]
    )
    if keep.sum() < 2:
        return float("nan"), float("nan"), np.array([])

    observed = float(adjusted_rand_score(labels_stage[keep], labels_cluster[keep]))
    rng = np.random.default_rng(seed)
    active_cluster = labels_cluster[keep]
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(active_cluster)
        null[i] = adjusted_rand_score(labels_stage[keep], shuffled)
    empirical_p = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return observed, empirical_p, null


def sensitivity_sweep(
    modality: str = "BAS",
    min_cluster_sizes: Tuple[int, ...] = (15, 30, 60, 120),
    seed: int = 42,
    embedding_dir: Optional[Path] = None,
    xml_dir: Optional[Path] = None,
    umap_cluster_dim: int = 15,
    umap_n_neighbors: int = 5,
    umap_metric: str = "cosine",
) -> pd.DataFrame:
    """HDBSCAN min_cluster_size sweep on chunk-level clustering.

    Small min_cluster_size can inflate subject-purity by creating many
    single-patient microclusters; large values collapse structure. Report
    n_clusters / n_noise / stage ARI / subject purity per setting so the
    user can pick a defensible value or show robustness across a range.

    Loads embeddings + XML once and fits UMAP once — HDBSCAN itself is the
    only per-iteration cost (audit finding 6). Previously each mcs value
    re-ran the whole `run_chunk_analysis` including HDF5 reads and UMAP
    fit, which for large cohorts wasted the majority of the runtime.
    """
    import umap
    import hdbscan
    from sklearn.metrics import silhouette_score

    # Load + label once
    X, subject_ids = load_chunk_embeddings(embedding_dir=embedding_dir, modality=modality)
    chunks_per_subject: Dict[str, int] = {}
    for sid in subject_ids:
        chunks_per_subject[sid] = chunks_per_subject.get(sid, 0) + 1
    xml_dir = Path(xml_dir) if xml_dir else resolve_cohort().xml_dir
    stages_by_subject, _ = load_chunk_labels_for_cohort(xml_dir, chunks_per_subject)
    stage_labels = _align_labels_to_chunks(subject_ids, stages_by_subject)

    # Fit UMAP once — same seed + same input -> identical output across the sweep
    n = X.shape[0]
    reducer = umap.UMAP(
        n_components=min(umap_cluster_dim, max(2, n - 2)),
        n_neighbors=min(umap_n_neighbors, max(2, n - 1)),
        min_dist=0.0, metric=umap_metric, random_state=seed,
    )
    cluster_coords = reducer.fit_transform(X)

    rows = []
    for mcs in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=1, metric="euclidean",
        )
        cluster_labels = clusterer.fit_predict(cluster_coords)
        # Wrap in a lightweight result for reuse of purity + ARI helpers.
        result = ChunkAnalysisResult(
            modality=modality, X=X, subject_ids=subject_ids,
            stage_labels=stage_labels, cluster_labels=cluster_labels,
            umap_2d=np.zeros((n, 2)),  # not needed for sweep metrics
        )
        summary = cluster_size_summary(result)
        purity = cluster_subject_purity(result)
        rows.append({
            "min_cluster_size": mcs,
            "n_clusters": summary["n_clusters"],
            "n_noise": summary["n_noise"],
            "stage_ari": stage_ari(result),
            "mean_subject_purity": float(purity["top_subject_share"].mean())
                                    if not purity.empty else float("nan"),
        })
    return pd.DataFrame(rows)


def cluster_size_summary(result: ChunkAnalysisResult) -> Dict[str, int]:
    summary = cluster_summary(result.cluster_labels)
    return {
        "n_chunks": len(result.cluster_labels),
        "n_clusters": len([k for k in summary if k >= 0]),
        "n_noise": summary.get(-1, 0),
        "distribution": {k: v for k, v in summary.items() if k >= 0},
    }

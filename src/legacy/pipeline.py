from datetime import datetime
import numpy as np
from typing import Optional
import hdbscan as hdbscan_lib
from sklearn.neighbors import NearestNeighbors

from config import MODALITY_DIM_RANGES, DISEASE_NAMES
from similarity_engine import build_reference_index, query_top5
from utils import normalize_l2


def compute_ablation_scores(
    query_embedding: np.ndarray,
    reference_index: NearestNeighbors,
    reference_labels: np.ndarray,
) -> dict:
    """Compute modality importance via mask-based ablation.

    For each modality, zero-out its dimensions in the query embedding,
    re-normalize to unit length, and measure the KNN similarity drop
    compared to the baseline (unmasked) query.

    Larger drop = modality is more important for this patient's similarity.

    Args:
        query_embedding: L2-normalized patient embedding, shape (128,).
        reference_index: Pre-built NearestNeighbors index.
        reference_labels: Labels for reference patients.

    Returns:
        Dict mapping modality name to similarity drop (float >= 0).
    """
    # Baseline: top-1 similarity with unmasked embedding
    baseline_results = query_top5(
        query_embedding=query_embedding,
        index=reference_index,
        reference_labels=reference_labels,
        disease_names=DISEASE_NAMES,
    )
    baseline_similarity = baseline_results[0]["similarity"] if baseline_results else 0.0

    scores = {}
    for modality, (start, end) in MODALITY_DIM_RANGES.items():
        # Zero-out this modality's dimensions
        ablated = query_embedding.copy()
        ablated[start:end] = 0.0

        # Re-normalize to unit length
        norm = np.linalg.norm(ablated)
        if norm > 0:
            ablated = ablated / norm
        else:
            # All dimensions were in this modality — max importance
            scores[modality] = round(baseline_similarity, 4)
            continue

        # Measure similarity with ablated embedding
        ablated_results = query_top5(
            query_embedding=ablated,
            index=reference_index,
            reference_labels=reference_labels,
            disease_names=DISEASE_NAMES,
        )
        ablated_similarity = ablated_results[0]["similarity"] if ablated_results else 0.0

        # Similarity drop: how much did removing this modality hurt?
        drop = max(0.0, baseline_similarity - ablated_similarity)
        scores[modality] = round(drop, 4)

    return scores


def run_pipeline(
    query_embedding: np.ndarray,          # (128,) — tek hasta
    reference_embeddings: np.ndarray,     # (n_ref, 128)
    reference_labels: np.ndarray,         # (n_ref,)
    umap_reducer,                         # fit edilmiş UMAP nesnesi
    hdbscan_model,                        # fit edilmiş HDBSCAN nesnesi
    n_neighbors: int = 10,
    patient_id: str = "unknown",
    reference_index: Optional[NearestNeighbors] = None,
) -> dict:
    """Tek hasta için tam pipeline. JSON-serializable dict döndürür.

    Adımlar:
    1. query_embedding L2-norm hesapla (sanity check: ~1.0 olmalı)
    2. UMAP ile 2D'ye indir (umap_reducer.transform)
    3. HDBSCAN ile küme ataması (hdbscan_lib.approximate_predict)
    4. Similarity Engine ile Top-5 hastalık
    5. Modality scores hesapla (ablation-based: her modality sıfırlanıp similarity drop ölçülür)
    6. JSON-serializable dict döndür
    """
    # Adım 1: L2-norm sanity check
    embedding_norm = float(np.linalg.norm(query_embedding))

    # Adım 2: UMAP 2D projeksiyon
    umap_coords_2d = umap_reducer.transform(query_embedding.reshape(1, -1))  # (1, 2)
    umap_coords = [round(float(umap_coords_2d[0, 0]), 4),
                   round(float(umap_coords_2d[0, 1]), 4)]

    # Adım 3: HDBSCAN küme ataması
    cluster_labels, _ = hdbscan_lib.approximate_predict(
        hdbscan_model, umap_coords_2d
    )
    cluster_id = int(cluster_labels[0])

    # Adım 4: Similarity Engine — Top-5 hastalık
    if reference_index is None:
        reference_index = build_reference_index(reference_embeddings, n_neighbors=n_neighbors)
    top5 = query_top5(
        query_embedding=query_embedding,
        index=reference_index,
        reference_labels=reference_labels,
        disease_names=DISEASE_NAMES,
    )

    # Adım 5: Modality scores via ablation-based attribution
    modality_scores = compute_ablation_scores(
        query_embedding=query_embedding,
        reference_index=reference_index,
        reference_labels=reference_labels,
    )

    # Adım 6: JSON-serializable dict
    return {
        "patient_id": patient_id,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "embedding_norm": round(embedding_norm, 4),
        "umap_coords": umap_coords,
        "cluster_id": cluster_id,
        "top5_diseases": top5,
        "modality_scores": modality_scores,
    }

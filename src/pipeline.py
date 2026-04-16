from datetime import datetime
import numpy as np
from typing import Optional
import hdbscan as hdbscan_lib

from config import MODALITY_DIM_RANGES, DISEASE_NAMES
from similarity_engine import build_reference_index, query_top5


def run_pipeline(
    query_embedding: np.ndarray,          # (128,) — tek hasta
    reference_embeddings: np.ndarray,     # (n_ref, 128)
    reference_labels: np.ndarray,         # (n_ref,)
    umap_reducer,                         # fit edilmiş UMAP nesnesi
    hdbscan_model,                        # fit edilmiş HDBSCAN nesnesi
    n_neighbors: int = 10,
    patient_id: str = "unknown",
) -> dict:
    """Tek hasta için tam pipeline. JSON-serializable dict döndürür.

    Adımlar:
    1. query_embedding L2-norm hesapla (sanity check: ~1.0 olmalı)
    2. UMAP ile 2D'ye indir (umap_reducer.transform)
    3. HDBSCAN ile küme ataması (hdbscan_lib.approximate_predict)
    4. Similarity Engine ile Top-5 hastalık
    5. Modality scores hesapla (her modalite boyutlarının ortalama abs değeri)
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
    index = build_reference_index(reference_embeddings, n_neighbors=n_neighbors)
    top5 = query_top5(
        query_embedding=query_embedding,
        index=index,
        reference_labels=reference_labels,
        disease_names=DISEASE_NAMES,
    )

    # Adım 5: Modality scores
    modality_scores = {}
    for modality, (start, end) in MODALITY_DIM_RANGES.items():
        modality_scores[modality] = round(
            float(np.abs(query_embedding[start:end]).mean()), 4
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

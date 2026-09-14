from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Dict, List
from collections import Counter


def build_reference_index(
    embeddings: np.ndarray,
    n_neighbors: int = 10,
) -> NearestNeighbors:
    """KNN index'i eğit. Dönen nesne query_top5()'e paslanır.

    Args:
        embeddings: L2-normalized reference embeddings, shape (n_samples, 128).
        n_neighbors: Sorgu başına bakılacak komşu sayısı.

    Returns:
        Fit edilmiş NearestNeighbors nesnesi.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")
    nn.fit(embeddings)
    return nn


def query_top5(
    query_embedding: np.ndarray,
    index: NearestNeighbors,
    reference_labels: np.ndarray,
    disease_names: Dict[int, str],
) -> List[Dict]:
    """K en yakın komşu üzerinden label voting ile Top-5 hastalık döndür.

    Algoritma:
    1. KNN index'ten K komşu bul (mesafeler + indeksler)
    2. Komşuların etiketlerini al
    3. Her etiket için: oy say (n_votes) ve o etiketteki komşuların
       ortalama mesafesinden similarity = 1 - (mean_distance / 2) hesapla
    4. n_votes'a göre azalan sırada sırala (eşit oylarda similarity bozar)
    5. Top-5'i döndür

    Returns:
        [{"rank": 1, "disease": "Heart Failure", "similarity": 0.87, "n_votes": 4}, ...]
    """
    distances, indices = index.kneighbors(query_embedding.reshape(1, -1))
    distances = distances[0]   # (n_neighbors,)
    indices = indices[0]       # (n_neighbors,)

    neighbor_labels = reference_labels[indices]

    # Her etiket için oy sayısı ve ortalama mesafe hesapla
    label_votes: Counter = Counter(neighbor_labels)
    label_distances: Dict[int, list] = {}
    for label, dist in zip(neighbor_labels, distances):
        label_distances.setdefault(label, []).append(dist)

    label_similarity = {
        label: 1.0 - (float(np.mean(dists)) / 2.0)
        for label, dists in label_distances.items()
    }

    # n_votes azalan, eşitlik bozucu olarak similarity azalan sırada sırala
    sorted_labels = sorted(
        label_votes.keys(),
        key=lambda lbl: (label_votes[lbl], label_similarity[lbl]),
        reverse=True,
    )

    results = []
    for rank, label in enumerate(sorted_labels[:5], start=1):
        results.append({
            "rank": rank,
            "disease": disease_names.get(int(label), f"Unknown({label})"),
            "similarity": round(label_similarity[label], 4),
            "n_votes": int(label_votes[label]),
        })

    return results

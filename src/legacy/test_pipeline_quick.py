import sys
sys.path.insert(0, 'src')

import numpy as np
from mock_data import generate_mock_embeddings
from similarity_engine import build_reference_index
import umap, hdbscan, json
from pipeline import run_pipeline

embeddings, labels = generate_mock_embeddings(seed=42, n_samples=500)
print(f"embeddings: {embeddings.shape}, labels: {labels.shape}")

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_2d = reducer.fit_transform(embeddings)
print(f"UMAP fit done: {umap_2d.shape}")

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
clusterer.fit(umap_2d)
n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
print(f"HDBSCAN: {n_clusters} clusters")

results = []
for i in [0, 100, 200]:
    # HATA #4 fix: exclude query from reference to avoid self-reference
    mask = np.ones(len(embeddings), dtype=bool)
    mask[i] = False
    ref_embeddings = embeddings[mask]
    ref_labels = labels[mask]
    ref_index = build_reference_index(ref_embeddings, n_neighbors=10)
    r = run_pipeline(
        query_embedding=embeddings[i],
        reference_embeddings=ref_embeddings,
        reference_labels=ref_labels,
        umap_reducer=reducer,
        hdbscan_model=clusterer,
        reference_index=ref_index,
        patient_id=f"mock_{i:03d}",
    )
    results.append(r)
    print(f"\n{r['patient_id']} | norm={r['embedding_norm']} | cluster={r['cluster_id']}")
    print(f"  top1: {r['top5_diseases'][0]}")
    print(f"  modality_scores: {r['modality_scores']}")

# JSON serializability check
serialized = json.dumps(results)
loaded = json.loads(serialized)
assert len(loaded) == 3

# Save to outputs/
import os
os.makedirs("outputs", exist_ok=True)
with open("outputs/pipeline_demo.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nJSON OK — outputs/pipeline_demo.json saved")

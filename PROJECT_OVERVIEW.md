# SleepFM Interpretability Project — Overview & File Navigation Guide

> **⚠️ Bu belge Nisan 2026'da yazıldı, mock-data paradigmasını anlatır.**
> Proje Eylül 2026'dan itibaren gerçek MESA cohort'una taşındı.
> Güncel durum ve mimari için: [README.md](README.md).
> Güncel bilimsel özet: [reports/phase15_defensibility/README.md](reports/phase15_defensibility/README.md).
> Aşağıdaki içerik UMAP/HDBSCAN/KNN kavramsal eğitim değeri için korunuyor —
> "everything runs on synthetic data" ifadesi artık geçersiz.

---

## What Is This Project?

**SleepFM** is a multimodal foundation model trained on sleep study data.
It produces 128-dimensional patient embeddings from four biosignal modalities:
**EEG** (brain), **ECG** (heart), **Respiratory**, and **EMG** (muscle).

This project builds an **interpretability pipeline** on top of those embeddings:
given a patient's embedding, it answers:
- *Which disease group is this patient most similar to?*
- *Which biosignal modality is driving the prediction?*
- *Where does this patient sit in the embedding space?*

**Nisan 2026 zeminindeki durum (bu belge yazıldığında):** Pipeline mock/sentetik
veri üzerinde tasarlanmış ve doğrulanmıştı. **Eylül 2026 itibarıyla** pipeline
gerçek SleepFM embedding'leri + 20 MESA hastası üzerinde çalışıyor
(Phase 13c ve sonrası). Mock katmanı legacy olarak korunuyor,
bilimsel çıkarım için kullanılmıyor.

---

## Architecture in One Diagram

```
Patient Embedding (128-dim, L2-normalized)
        │
        ▼
   ┌─────────┐        ┌───────────────┐
   │  UMAP   │──2D──▶│  HDBSCAN      │──▶ cluster_id (int)
   │ reducer │        │  approximate_ │
   └─────────┘        │  predict      │
                      └───────────────┘
        │
        ▼
   ┌──────────────────────┐
   │  Similarity Engine   │──▶ Top-5 diseases + similarity scores
   │  (KNN, k=10,         │
   │   euclidean metric)  │
   └──────────────────────┘
        │
        ▼
   Modality Scores  (mean |value| per modality dimension block)
   eeg:[0–31]  ecg:[32–63]  resp:[64–95]  emg:[96–127]
        │
        ▼
   JSON Output  {patient_id, umap_coords, cluster_id,
                 top5_diseases, modality_scores, ...}
```
## NOTE 1:

## UMAP (Uniform Manifold Approximation and Projection)

**Ne işe yarar:** Yüksek boyutlu veriyi (örn. 128-dim embedding) 2 boyuta indirger; veriyi görselleştirilebilir hale getirir.

- Uzayda birbirine yakın olan noktaların düşük boyutta da yakın kalmasını sağlar.
- Bu projede: 128-dim hasta embedding'lerini → 2D koordinata dönüştürür.
- **`fit`** → referans popülasyon üzerinde manifold'u öğrenir.
- **`transform`** → yeni hasta geldiğinde o öğrenilmiş manifold'a yerleştirir (yeniden öğrenmez).

---

## HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

**Ne işe yarar:** 2D UMAP uzayındaki yoğunluk bölgelerini otomatik olarak kümelere ayırır; gürültülü/sınır noktaları `-1` (noise) olarak işaretler.

- Küme sayısını önceden belirtmek gerekmez, veriden öğrenir.
- Bu projede: UMAP çıktısındaki 2D noktaları → `cluster_id` (tamsayı(CLUSTER=KÜME)) etiketine atar.
- **`fit`** → referans popülasyon üzerinde küme sınırlarını öğrenir.
- **`approximate_predict`** → yeni hasta için en yakın kümeyi tahmin eder (yeniden eğitmez).

---

> **İkisi birlikte:** UMAP boyutu düşürür → HDBSCAN o düşük boyutlu uzayda kümeleri bulur.
> Pipeline'da her ikisi de **önceden eğitilmiş** olarak gelir; yeni hasta sadece bu sabit manifold üzerine **yerleştirilir**.

## NOTE2: K-Nearest Neighbors (KNN)

KNN is considered "different" from most machine learning algorithms because it is a **lazy learner** and **non-parametric**, meaning it does not build an explicit model or learn internal parameters during a training phase. Instead, it memorizes the entire dataset and performs all computations only at the moment of prediction.

---

### Why KNN Is Different — Key Characteristics

**1. Lazy Learning (No Training Phase)**

Unlike (aksine) "eager" algorithms (e.g., Linear Regression, Neural Networks) that create a formula or internal representation from data, KNN does not "learn" anything upfront.

- **How it works:** When you give it training data, it simply stores it.
- **The Difference:** The computational heavy lifting is deferred until you ask for a prediction (inference time). This makes training instantaneous (anlık) but prediction slow, especially with large datasets.

**2. Non-Parametric**

KNN makes no assumptions (varsayımda bulunmak) about the underlying distribution (dağılım) of the data.

- **The Difference:** It doesn't assume the data fits a straight line (linear) or a specific distribution (like Gaussian(in two lower lines)). This makes it highly flexible for complex, non-linear data.

- EKSTRA INFO: The Gaussian distribution, or normal distribution, is a symmetric, bell-shaped probability distribution that describes how real-valued random variables are spread around a mean.

**3. Instance-Based (Memory-Based)**

KNN is entirely dependent on the training data to make predictions.

- **The Difference:** The model *is* the data. Because it must calculate the distance between a new point and every single stored training point, its memory usage grows linearly with the amount of data, making it inefficient for very large datasets.

**4. Highly Sensitive to Feature Scaling and Noise**

Because KNN relies on distance metrics (like Euclidean distance(in two lower lines)) to determine proximity (yakınlık), it is highly sensitive to the scale of features.
- EKSTRA INFO: Euclidean distance is the straight-line distance between two points in Euclidean space, commonly known as the "as-the-crow-flies" distance. It is calculated using the Pythagorean theorem, representing the shortest path, and is widely used in machine learning (KNN, K-Means) and data analysis to measure similarity.

- **The Difference:** If one feature is in millions and another is a percentage (0–1), the larger feature will dominate (hakim olmak) the distance calculation. Data normalization or standardization is mandatory (zorunlu). It is also sensitive to outliers, which can easily mislead (yanıltmak) a classification.

---

### Summary: KNN vs. Eager Learners

| Feature          | KNN (Lazy)           | Neural Net / Decision Tree (Eager) |
|------------------|----------------------|------------------------------------|
| Training Time    | None (Stores data)   | High (Iterative learning)          |
| Prediction Time  | High (Calculates distances) | Low (Uses model formula)    |
| Memory Usage     | High (Stores all data) | Low (Stores only model parameters)|
| Best For         | Small, clean datasets | Large, complex datasets           |

> **Note:** KNN is frequently confused with K-Means, but they are fundamentally different: KNN is **supervised** (classification), while K-Means is **unsupervised** (clustering).

---

## Repository Structure

```
sleepfm_interpretability/
│
├── src/                          ← All production Python code
│   ├── config.py                 ← Single source of truth (diseases, dims, constants)
│   ├── mock_data.py              ← Synthetic embedding generator
│   ├── similarity_engine.py     ← KNN index + Top-5 query logic
│   ├── pipeline.py              ← End-to-end single-patient pipeline
│   ├── visualization.py         ← UMAP scatter + Modality heatmap (PNG)
│   ├── metrics.py               ← C-Index, AUROC, bootstrap CI
│   ├── utils.py                 ← Seeds, L2-normalize, timer decorator
│   └── __init__.py
│
├── tests/
│   ├── test_embedding_generation.py   ← 20 pytest tests (all passing)
│   └── __init__.py
│
├── notebooks/
│   ├── 01_hdbscan_practice.ipynb      ← UMAP + HDBSCAN exploration
│   └── 02_pipeline_demo.ipynb         ← End-to-end demo (3 patients, JSON + plots)
│
├── outputs/                      ← Generated PNGs (git-ignored)
│   ├── mock_umap.png             ← UMAP scatter of 500 patients × 12 disease groups
│   ├── mock_modality_importance.png   ← 12×4 heatmap (disease × modality)
│   └── pipeline_demo.json        ← 3-patient pipeline output
│
├── practice/
│   └── KARAR_NOKTALARI.md        ← Design decision log (Turkish)
│
├── conftest.py                   ← pytest path config (adds src/ to sys.path)
└── requirements.txt              ← Python dependencies
```

---

## File-by-File Guide

### `src/config.py` — Start Here If You Want to Understand the Data Model

Defines every constant that the rest of the code uses.
The most important things:

| Symbol | Value | Meaning |
|--------|-------|---------|
| `EMBEDDING_DIM` | 128 | Total embedding dimension |
| `MODALITY_DIM_RANGES` | `{"eeg":(0,32), "ecg":(32,64), "resp":(64,96), "emg":(96,128)}` | Which dims belong to which modality |
| `DISEASES` | dict of 12 `DiseaseConfig` objects | Heart Failure, AFib, Stroke, ... |
| `DISEASE_NAMES` | `{0: "Heart Failure", 1: "Atrial Fibrillation", ...}` | Label int → name |

**Key design decision:** Equal 32-dim split across 4 modalities avoids any
implicit weighting from dimension count imbalance.

---

### `src/mock_data.py` — How the Synthetic Data Is Made

```python
embeddings, labels = generate_mock_embeddings(seed=42, n_samples=500)
# embeddings: (500, 128), L2-normalized
# labels: (500,), integers 0–11 (one per disease group)
```

**How it works:**
1. Creates 12 cluster centers, each with a different modality activation pattern
   (e.g., cardiac diseases have higher magnitude in ECG dims 32–63)
2. Generates samples around centers with per-modality Gaussian noise
3. Randomly reassigns 15% of samples to wrong clusters (intentional overlap)

**Why intentional overlap?** Perfect clustering would be scientifically wrong —
real patients have comorbidities and mixed signals.

Other functions:
- `generate_mock_coxph_scores(labels)` — synthetic survival risk scores
- `build_disease_weight_profiles()` — per-disease 128-dim weight vectors (for heatmap)

---

### `src/similarity_engine.py` — The Core Recommendation Logic

Two functions:

```python
# Step 1: Build index once from all reference patients
index = build_reference_index(embeddings, n_neighbors=10)

# Step 2: Query for a new patient
results = query_top5(query_embedding, index, labels, DISEASE_NAMES)
# → [{"rank":1, "disease":"Heart Failure", "similarity":0.87, "n_votes":4}, ...]
```

**Algorithm for `query_top5`:**
1. Find k=10 nearest neighbors (Euclidean distance on L2-normalized vectors ≡ cosine similarity)
2. Count votes per disease label among the 10 neighbors
3. Compute `similarity = 1 − (mean_distance / 2)` → normalized to [0, 1]
4. Sort by vote count (ties broken by similarity), return top 5

**Why Euclidean on L2-normalized vectors?**
On unit-norm vectors, Euclidean² = 2 − 2·cosine_similarity.
Ranking is equivalent; sklearn's euclidean is faster than cosine kernel.

---

### `src/pipeline.py` — The Main Entry Point

```python
result = run_pipeline(
    query_embedding=embeddings[i],       # (128,) for one patient
    reference_embeddings=embeddings,     # (500, 128) reference pool
    reference_labels=labels,             # (500,) disease labels
    umap_reducer=fitted_umap,            # pre-fitted UMAP object
    hdbscan_model=fitted_hdbscan,        # pre-fitted HDBSCAN object
    patient_id="patient_001",
)
```

**Output JSON structure:**
```json
{
  "patient_id": "mock_000",
  "timestamp": "2026-04-16T14:23:00",
  "embedding_norm": 1.0,
  "umap_coords": [1.23, -0.84],
  "cluster_id": 3,
  "top5_diseases": [
    {"rank": 1, "disease": "Heart Failure", "similarity": 0.87, "n_votes": 4},
    ...
  ],
  "modality_scores": {
    "eeg": 0.65, "ecg": 0.89, "resp": 0.72, "emg": 0.41
  }
}
```

**`modality_scores` interpretation:** Mean absolute value of the embedding in each
modality's dimension block. High ECG score → patient's signal is cardiac-dominant.

**Important:** UMAP and HDBSCAN models must be **pre-fitted** on the reference
population and passed in. The pipeline calls `umap_reducer.transform()` (not fit)
and `hdbscan.approximate_predict()` (not fit) — this is by design, so the
manifold is fixed and consistent across queries.

---

### `src/visualization.py` — Static PNG Outputs

```python
# UMAP scatter: 12 disease groups in different colors
plot_umap_scatter(umap_2d, labels, DISEASE_NAMES,
                  highlight_idx=42,          # mark this patient with a star
                  save_path="outputs/mock_umap.png")

# Modality importance heatmap: 12 diseases × 4 modalities
plot_modality_importance_heatmap(
    weight_matrix,                           # (12, 4) numpy array
    disease_names=disease_list,
    modality_names=["EEG", "ECG", "Resp", "EMG"],
    save_path="outputs/mock_modality_importance.png")
```

Expected visual patterns:
- Cardiac diseases (Heart Failure, AFib) → dark red ECG column
- Neurological diseases (Dementia, Stroke, Depression) → dark red EEG column

---

### `src/metrics.py` — Statistical Evaluation (Not Used in Pipeline Yet)

Ready for when real SleepFM embeddings arrive:
- `concordance_index(event_times, predicted_scores, event_indicators)` → C-Index
- `auroc(y_true, y_score)` → AUROC
- `bootstrap_ci(metric_fn, ...)` → 95% confidence intervals via percentile bootstrap
- `permutation_test(metric_fn, ...)` → p-value via label permutation

---

### `src/utils.py` — Small Helpers

- `set_all_seeds(seed)` — numpy + random + os.environ seeds
- `normalize_l2(x)` — row-wise L2 normalization of a 2D array
- `timer` — decorator that prints execution time
- `validate_noise_config(config)` — raises ValueError for bad noise configs

---

### `tests/test_embedding_generation.py` — 20 Tests, All Passing

Test classes:
| Class | What it tests |
|-------|---------------|
| `TestEmbeddingShape` | Output shapes and label counts |
| `TestEmbeddingNormalization` | Unit norm, no NaN/inf |
| `TestClusterStructure` | Intra-cluster < inter-cluster distance |
| `TestPerturbation` | Perturbation rate behavior |
| `TestNoiseConfig` | Per-modality noise effects |
| `TestReproducibility` | Same seed → same output |
| `TestValidation` | Edge case error handling |

Run with: `python -m pytest tests/ -v`

---

### `notebooks/02_pipeline_demo.ipynb` — The End-to-End Demo

**Best place to see the full pipeline in action.**

Cell sequence:
1. Imports + sys.path setup
2. Generate 500 mock embeddings
3. Fit UMAP (n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
4. Fit HDBSCAN (min_cluster_size=15, prediction_data=True)
5. Run `run_pipeline()` for 3 patients (indices 0, 100, 200)
6. Save JSON to `outputs/pipeline_demo.json`
7. Plot UMAP scatter with patient 0 highlighted
8. Plot modality importance heatmap

---

### `notebooks/01_hdbscan_practice.ipynb` — Learning Notebook

UMAP + HDBSCAN experiments on mock data:
- Effect of `min_cluster_size` on cluster count
- UMAP visualization of 12 disease groups
- Saved as `outputs/mock_umap.png`

---

## How to Run Everything From Scratch

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all tests
python -m pytest tests/ -v

# 3. Run the full pipeline demo
# Open notebooks/02_pipeline_demo.ipynb and run all cells

# 4. Or run a quick CLI smoke test
python -c "
import sys; sys.path.insert(0, 'src')
from mock_data import generate_mock_embeddings
from pipeline import run_pipeline
import umap, hdbscan, json

embeddings, labels = generate_mock_embeddings(seed=42)
reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
umap_2d = reducer.fit_transform(embeddings)
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
clusterer.fit(umap_2d)

result = run_pipeline(embeddings[0], embeddings, labels, reducer, clusterer)
print(json.dumps(result, indent=2))
"
```

---

## Key Design Decisions (Short Version)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding dim split | Equal 32×4 | No implicit modality weighting |
| Distance metric | Euclidean on L2-norm | Equivalent to cosine, faster in sklearn |
| KNN k value | 10 | Balanced: stable vote counts, not overkill for n=500 |
| Bootstrap | Manual (numpy) | Transparency > library convenience |
| UMAP+HDBSCAN | Pre-fit, transform only | Fixed manifold across queries |
| Mock data overlap | 15% perturbation | Realistic comorbidity simulation |

Full decision log: `practice/KARAR_NOKTALARI.md`

---

## What's Next (Day 7 Roadmap)

- [ ] `tests/test_similarity_engine.py` — at least 5 tests for KNN logic
- [ ] `tests/test_pipeline.py` — JSON schema validation, edge cases
- [ ] Search SleepFM `pretrain.py` for `InfoNCE` / `contrastive` to understand
      the actual training loop (Leave-One-Out Contrastive Learning)
- [ ] Replace mock embeddings with real SleepFM outputs when available

---

*Generated: 2026-04-16 | Project: SleepFM Interpretability | Author: Aslı Aktaş*

# SleepFM Interpretability

Embedding analysis, similarity engine, and pipeline integration for the SleepFM foundation model.

---

## 🔥 Project Updates

### ✅ Completed Modules

- `src/similarity_engine.py` — KNN-based Top-5 disease recommendation
- `tests/` — 20 pytest tests (all passing)
- `notebooks/01_hdbscan_practice.ipynb` — UMAP visualization

### 🚧 In Progress

- `src/pipeline.py` — End-to-end inference pipeline
- `src/visualization.py` — UMAP scatter and heatmap visualization
- `notebooks/02_pipeline_demo.ipynb` — Pipeline demo notebook

---

## 📖 Introduction

SleepFM is a multimodal sleep foundation model trained to generate disease risk predictions from polysomnography (PSG) recordings. This project aims to analyze the 128-dimensional embedding vectors produced by SleepFM, perform clustering, and provide clinical interpretability via a KNN-based disease similarity engine.

The embedding space is divided into 4 modalities:

| Modality | Dims | Description |
|---|---|---|
| **EEG** | 0–31 | Electroencephalography |
| **ECG** | 32–63 | Electrocardiography |
| **Resp** | 64–95 | Respiratory signals |
| **EMG** | 96–127 | Electromyography |

---

## 📖 Table of Contents

1. [Installation](#-installation)
2. [Project Structure](#-project-structure)
3. [Usage](#-usage)
4. [Tests](#-tests)
5. [Technical Details](#-technical-details)
6. [References](#-references)

---

## 💿 Installation

### 🖥️ Requirements

- Python 3.10+
- scikit-learn, numpy, umap-learn, hdbscan, matplotlib

### 🚀 Setup

```bash
git clone https://github.com/AsliAktas/sleepfm_interpretability.git
cd sleepfm_interpretability
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
sleepfm_interpretability/
├── src/
│   ├── config.py              # Disease names and constants
│   ├── mock_data.py           # Mock embedding generation (500 patients, 12 diseases)
│   ├── metrics.py             # Evaluation metrics
│   ├── similarity_engine.py   # KNN-based Top-5 disease recommendation
│   └── utils.py               # Helper functions
├── tests/
│   └── test_embedding_generation.py  # 20 pytest tests
├── notebooks/
│   └── 01_hdbscan_practice.ipynb     # UMAP visualization
├── practice/
│   └── KARAR_NOKTALARI.md
├── conftest.py                # pytest path configuration
├── requirements.txt
└── README.md
```

---

## 👩‍💻 Usage

### Mock Data Generation

```python
from src.mock_data import generate_mock_embeddings

embeddings, labels = generate_mock_embeddings(seed=42, n_samples=500)
# embeddings: (500, 128) — L2-normalized, 4×32 modalities (EEG/ECG/Resp/EMG)
# labels: (500,) — 12 disease classes
```

### Similarity Engine

```python
from src.similarity_engine import build_reference_index, query_top5
from src.config import DISEASE_NAMES

index = build_reference_index(embeddings, n_neighbors=10)
results = query_top5(embeddings[0], index, labels, DISEASE_NAMES)
# [{'rank': 1, 'disease': 'Obesity', 'similarity': 0.7838, 'n_votes': 7}, ...]
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

20 tests, all passing ✅

| Test Class | Coverage |
|---|---|
| `TestEmbeddingShape` | Shape validation |
| `TestEmbeddingNormalization` | L2-norm check |
| `TestClusterStructure` | Cluster structure |
| `TestPerturbation` | Perturbation rate |
| `TestNoiseConfig` | Noise configuration |
| `TestReproducibility` | Reproducibility |
| `TestValidation` | Input validation |

---

## ⚙️ Technical Details

| Parameter | Value |
|---|---|
| Embedding size | 128-dim |
| Normalization | L2-normalized |
| KNN metric | Euclidean |
| Similarity formula | `1 - (mean_distance / 2)` → [0, 1] |
| Number of diseases | 12 |

---

## 📚 References

- **SleepFM**: [A multimodal sleep foundation model for disease prediction](https://doi.org/10.1038/s41591-025-04133-4) — *Nature Medicine, 2026*
- **GitHub**: [zou-group/sleepfm-clinical](https://github.com/zou-group/sleepfm-clinical)

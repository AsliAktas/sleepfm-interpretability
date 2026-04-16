# SleepFM Interpretability

Sleep foundation model embedding analizi, similarity engine ve pipeline entegrasyonu projesi.

## Proje Yapısı

```
sleepfm_interpretability/
├── src/
│   ├── config.py              # Hastalık isimleri ve sabitler
│   ├── mock_data.py           # Mock embedding üretimi (500 hasta, 12 hastalık)
│   ├── metrics.py             # Değerlendirme metrikleri
│   ├── similarity_engine.py   # KNN tabanlı Top-5 hastalık önerisi
│   └── utils.py               # Yardımcı fonksiyonlar
├── tests/
│   └── test_embedding_generation.py  # 20 pytest testi
├── notebooks/
│   └── 01_hdbscan_practice.ipynb     # UMAP görselleştirme
├── practice/
│   └── KARAR_NOKTALARI.md
├── conftest.py                # pytest path konfigürasyonu
├── requirements.txt
└── README.md
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Mock Data Üretimi
```python
from src.mock_data import generate_mock_embeddings
embeddings, labels = generate_mock_embeddings(seed=42, n_samples=500)
# embeddings: (500, 128) — L2-normalized, 4×32 modalite (EEG/ECG/Resp/EMG)
# labels: (500,) — 12 hastalık sınıfı
```

### Similarity Engine
```python
from src.similarity_engine import build_reference_index, query_top5
from src.config import DISEASE_NAMES

index = build_reference_index(embeddings, n_neighbors=10)
results = query_top5(embeddings[0], index, labels, DISEASE_NAMES)
# [{'rank': 1, 'disease': 'Obesity', 'similarity': 0.7838, 'n_votes': 7}, ...]
```

## Testler

```bash
python -m pytest tests/ -v
# 20 test, hepsi geçiyor
```

## Teknik Detaylar

- **Embedding boyutu:** 128-dim (EEG: 0-31, ECG: 32-63, Resp: 64-95, EMG: 96-127)
- **Normalizasyon:** L2-normalized
- **KNN metrik:** euclidean (L2-normalized üzerinde cosine'e eşdeğer)
- **Similarity formülü:** `1 - (mean_distance / 2)` → [0, 1] aralığı
- **Hastalık sayısı:** 12

## İlgili Kaynaklar

- SleepFM: Foundation model for sleep EEG analysis
- ETHOS: Embedding-based clinical trial outcome prediction

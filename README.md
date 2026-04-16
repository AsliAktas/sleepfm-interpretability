# SleepFM Interpretability

Sleep foundation model embedding analizi, similarity engine ve pipeline entegrasyonu.

---

## 🔥 Proje Güncellemeleri

### ✅ Tamamlanan Modüller

- `src/similarity_engine.py` — KNN tabanlı Top-5 hastalık önerisi
- `tests/` — 20 pytest testi (hepsi geçiyor)
- `notebooks/01_hdbscan_practice.ipynb` — UMAP görselleştirme

### 🚧 Devam Eden Çalışmalar

- `src/pipeline.py` — Uçtan uca inference pipeline'ı
- `src/visualization.py` — UMAP scatter ve heatmap görselleştirme
- `notebooks/02_pipeline_demo.ipynb` — Pipeline demo notebook'u

---

## 📖 Giriş

SleepFM, polisomnografi (PSG) kayıtlarından hastalık risk tahminleri üretmek için eğitilmiş çok-modaliteli bir uyku foundation modelidir. Bu proje, SleepFM'in ürettiği 128-boyutlu gömme vektörlerini analiz etmeyi, kümeleme yapmayı ve KNN tabanlı hastalık benzerlik motoru ile klinik yorumlanabilirlik sağlamayı amaçlamaktadır.

Embedding uzayı 4 modaliteye bölünmüştür:

| Modalite | Dims | Açıklama |
|---|---|---|
| **EEG** | 0–31 | Elektroensefalografi |
| **ECG** | 32–63 | Elektrokardiyografi |
| **Resp** | 64–95 | Solunum sinyalleri |
| **EMG** | 96–127 | Elektromiyografi |

---

## 📖 İçindekiler

1. [Kurulum](#-kurulum)
2. [Proje Yapısı](#-proje-yapısı)
3. [Kullanım](#-kullanım)
4. [Testler](#-testler)
5. [Teknik Detaylar](#-teknik-detaylar)
6. [Kaynaklar](#-kaynaklar)

---

## 💿 Kurulum

### 🖥️ Gereksinimler

- Python 3.10+
- scikit-learn, numpy, umap-learn, hdbscan, matplotlib

### 🚀 Kurulum Adımları

```bash
git clone https://github.com/AsliAktas/sleepfm_interpretability.git
cd sleepfm_interpretability
pip install -r requirements.txt
```

---

## 📁 Proje Yapısı

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

---

## 👩‍💻 Kullanım

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

---

## 🧪 Testler

```bash
python -m pytest tests/ -v
```

20 test, hepsi geçiyor ✅

| Test Sınıfı | Kapsam |
|---|---|
| `TestEmbeddingShape` | Boyut doğrulama |
| `TestEmbeddingNormalization` | L2-norm kontrolü |
| `TestClusterStructure` | Küme yapısı |
| `TestPerturbation` | Pertürbasyon oranı |
| `TestNoiseConfig` | Gürültü konfigürasyonu |
| `TestReproducibility` | Tekrarlanabilirlik |
| `TestValidation` | Girdi doğrulama |

---

## ⚙️ Teknik Detaylar

| Parametre | Değer |
|---|---|
| Embedding boyutu | 128-dim |
| Normalizasyon | L2-normalized |
| KNN metrik | Euclidean |
| Similarity formülü | `1 - (mean_distance / 2)` → [0, 1] |
| Hastalık sayısı | 12 |

---

## 📚 Kaynaklar

- **SleepFM**: [A multimodal sleep foundation model for disease prediction](https://doi.org/10.1038/s41591-025-04133-4) — *Nature Medicine, 2026*
- **GitHub**: [zou-group/sleepfm-clinical](https://github.com/zou-group/sleepfm-clinical)

# GÜN 6 — Pipeline Entegrasyonu + Similarity Engine + Visualization

## BUGÜNÜN HEDEFİ
Gün sonunda elinde şunlar olacak:
1. Proje klasör yapısı düzenli: src/, tests/, notebooks/ hiyerarşisi kurulmuş
2. Similarity Engine tamamlanmış: KNN-based Top-5 hastalık tavsiyesi
3. Uçtan uca simüle pipeline çalışıyor: Embedding → UMAP → HDBSCAN → Similarity → JSON
4. visualization.py hazır: UMAP scatter plot + Modality Importance Matrix heatmap
5. ETHOS makalesi tamamlanmış, JMIR Aging başlatılmış

---

## GÜN 5'TEN DEVİR ALINAN GÖREVLER

| Görev | Durum | Nereye taşındı |
|---|---|---|
| Klasör yapısı (src/tests/notebooks) | ❌ Tamamlanmadı | Bugün 08:00–09:00 |
| Mock data UMAP görselleştirmesi | ❌ Tamamlanmadı | Bugün 09:00–09:30 |

---

## GÜN 5 HIZLI HATIRLATMA

### Gün 5 Çıktıları
- **mock_data.py**: `generate_mock_embeddings()`, `generate_mock_coxph_scores()`,
  `generate_mock_modality_embeddings()`, `generate_mock_survival_data()` hazır
- **config.py**: 12 hastalık tanımı, `MODALITY_DIM_RANGES`, `EMBEDDING_DIM=128`,
  istatistiksel sabitler (`BOOTSTRAP_N_RESAMPLES`, `PERMUTATION_N`, vb.)
- **metrics.py + metrics_skeleton.py**: C-Index wrapper, AUROC, `bootstrap_ci()`
  (metric_args destekli, concordance_index ile uyumlu), `permutation_test()`
- **utils.py**: `set_all_seeds()`, `timer` decorator, `normalize_l2()`, `validate_noise_config()`
- **test_embedding_generation.py**: 20 test, hepsi geçiyor (TestValidation dahil)
- **conftest.py**: pytest root path discovery
- **HDBSCAN pratiği**: anonim.ipynb'de blobs + min_cluster_size deneyleri

### Önemli Tasarım Kararları (Referans)
- **Karar 1**: 4×32 eşit boyut bölünmesi, modality-based dimension groups
- **Karar 4**: Manuel bootstrap, şeffaflık > kütüphane kolaylığı
- **Karar 6**: Multi-file class-based test organizasyonu
- **metrics_skeleton.py vs metrics.py**: İkisi birlikte tutuluyor.
  `metrics.py` = production dosyası. `metrics_skeleton.py` = öğrenme referansı/geçmiş.

---

## SAAT SAAT PROGRAM

### 08:00–09:00 | Klasör Reorganizasyonu (1 saat)

**Neden yapıyoruz:**
Şu an tüm dosyalar root'ta flat yapıda. Pipeline büyüdükçe flat yapıda module
import'ları karışır, test discovery güçleşir. Gün 5 rehberi src/tests/notebooks
yapısını öngörüyordu — bugün hayata geçiriyoruz.

**Hedef yapı:**
```
sleepfm_interpretability/
├── src/
│   ├── __init__.py          ← yeni (boş)
│   ├── mock_data.py         ← taşındı
│   ├── config.py            ← taşındı
│   ├── metrics.py           ← taşındı
│   ├── metrics_skeleton.py  ← taşındı
│   ├── utils.py             ← taşındı
│   ├── similarity_engine.py ← bugün yazılacak
│   ├── pipeline.py          ← bugün yazılacak
│   └── visualization.py     ← bugün yazılacak
├── tests/
│   ├── __init__.py          ← yeni (boş)
│   └── test_embedding_generation.py  ← taşındı
├── notebooks/
│   └── 01_hdbscan_practice.ipynb     ← taşındı (anonim.ipynb)
├── outputs/                 ← yeni (görsel çıktılar buraya)
├── conftest.py              ← root'ta KALIR (pytest bunu arar)
└── requirements.txt         ← root'ta KALIR
```

**Adım 1: Klasörleri oluştur**
```
src/, tests/, notebooks/, outputs/ klasörlerini oluştur.
src/__init__.py ve tests/__init__.py dosyalarını boş oluştur.
```

**Adım 2: conftest.py'yi güncelle**
Root'taki conftest.py artık `src/` dizinini de path'e eklemeli:
```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
```
Bu sayede `from mock_data import ...` şeklindeki import'lar src/ içindeki dosyaları bulur.

**Adım 3: Dosyaları taşı**
mock_data.py, config.py, metrics.py, metrics_skeleton.py, utils.py → src/
test_embedding_generation.py → tests/
anonim.ipynb → notebooks/01_hdbscan_practice.ipynb

**Adım 4: Test et — bu adım kritik**
```
python -m pytest tests/ -v
```
20 testin hepsi geçmeli. Geçmezse: conftest.py'deki sys.path eklemesini kontrol et.

---

### 09:00–09:30 | Mock Data UMAP Görselleştirmesi (30 dk) [Gün 5'ten taşındı]

notebooks/01_hdbscan_practice.ipynb'e yeni bir bölüm ekle:

```python
import umap
from mock_data import generate_mock_embeddings

embeddings, labels = generate_mock_embeddings(seed=42)

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_2d = reducer.fit_transform(embeddings)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
scatter = plt.scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels,
                      cmap="tab20", s=10, alpha=0.7)
plt.colorbar(scatter, label="Disease Group")
plt.title("Mock Embeddings — UMAP Projection (12 Disease Groups)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("../outputs/mock_umap.png", dpi=150)
plt.show()
```

**Beklenen çıktı:** 12 renk grubu, kısmen ayrışmış ama mükemmel olmayan kümeler.
Mükemmel ayrışma olsaydı mock data'nın rule #5'i ihlal ettiğini gösterirdi.

---

### 09:30–11:30 | Similarity Engine (2 saat)

**Kullanılacak kütüphane:** `sklearn.neighbors.NearestNeighbors`
(sklearn==1.3.2 zaten requirements.txt'te var)

**Tasarım Kararları:**

**K değeri:**
- K=5: Hızlı, az gürültü, ama küçük örneklemlerde kararsız
- K=10: Dengeli başlangıç, bu projeye uygun
- K=15: Daha stabil ama n_samples=500'de overkill
→ **K=10 varsayılan, parametrik (kullanıcı değiştirebilir)**

**Mesafe metriği:**
Embedding'ler L2-normalize edilmiş. Normalize vektörlerde:
- cosine_distance = 2 − 2·dot_product
- euclidean_distance² = 2 − 2·cosine_similarity
Matematiksel olarak **eşdeğer**. sklearn ile `metric="euclidean"` kullanmak yeterli.

**src/similarity_engine.py fonksiyon yapısı:**

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


def build_reference_index(
    embeddings: np.ndarray,       # (n_samples, 128) — referans embedding'ler
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
    query_embedding: np.ndarray,    # (128,) — tek hasta embedding'i
    index: NearestNeighbors,
    reference_labels: np.ndarray,   # (n_samples,) — index'e karşılık gelen etiketler
    disease_names: Dict[int, str],  # label → hastalık adı (config.DISEASE_NAMES)
) -> List[Dict]:
    """K en yakın komşu üzerinden label voting ile Top-5 hastalık döndür.

    Algoritma:
    1. KNN index'ten K komşu bul (mesafeler + indeksler)
    2. Komşuların etiketlerini al
    3. Her etiket için: oy say (n_votes) ve o etiketteki komşuların
       ortalama mesafesinden similarity = 1 - mean_distance hesapla
    4. n_votes'a göre azalan sırada sırala (eşit oylarda similarity bozar)
    5. Top-5'i döndür

    Returns:
        [{"rank": 1, "disease": "Heart Failure", "similarity": 0.87, "n_votes": 4}, ...]
    """
    ...
```

**Implementasyon Notu — similarity hesabı:**
Euclidean mesafe normalize vektörler üzerinde [0, 2] aralığında.
0 = aynı nokta (mükemmel eşleşme), 2 = zıt yönler.
`similarity = 1 - (mean_distance / 2)` → [0, 1] aralığına normalize eder.

---

### 11:30–13:00 | Pipeline Entegrasyonu + JSON Çıktı (1.5 saat)

**src/pipeline.py** — uçtan uca akış:

```python
from datetime import datetime
import json
import numpy as np
from typing import Optional
from sklearn.neighbors import NearestNeighbors

from config import MODALITY_DIM_RANGES, DISEASE_NAMES


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
    3. HDBSCAN ile küme ataması (hdbscan_model.approximate_predict veya yeni fit)
    4. Similarity Engine ile Top-5 hastalık
    5. Modality scores hesapla (her modalite boyutlarının ortalama abs değeri)
    6. JSON dict döndür
    """
    ...
```

**JSON çıktı formatı:**
```json
{
  "patient_id": "mock_001",
  "timestamp": "2026-04-16T14:23:00",
  "embedding_norm": 1.000,
  "umap_coords": [1.23, -0.84],
  "cluster_id": 3,
  "top5_diseases": [
    {"rank": 1, "disease": "Heart Failure",                    "similarity": 0.87, "n_votes": 4},
    {"rank": 2, "disease": "Atrial Fibrillation and Flutter",  "similarity": 0.79, "n_votes": 3},
    {"rank": 3, "disease": "Hypertension",                     "similarity": 0.71, "n_votes": 2},
    {"rank": 4, "disease": "Stroke",                           "similarity": 0.68, "n_votes": 1},
    {"rank": 5, "disease": "Depression",                       "similarity": 0.62, "n_votes": 0}
  ],
  "modality_scores": {
    "eeg":  0.65,
    "ecg":  0.89,
    "resp": 0.72,
    "emg":  0.41
  }
}
```

**`modality_scores` tanımı:** query embedding'in o modaliteye ait boyutlarının
(0–31, 32–63, 64–95, 96–127) ortalama absolute değeri.
Hangi modalitede sinyal güçlü — hızlı gösterge. Yorumlama:
ECG skoru yüksekse → hasta kardiyak profile yakın.

**Entegrasyon testi:**
```python
# notebooks/02_pipeline_demo.ipynb
# 3 mock hasta için pipeline çalıştır:
from mock_data import generate_mock_embeddings
embeddings, labels = generate_mock_embeddings(seed=42, n_samples=500)

# UMAP + HDBSCAN fit
import umap, hdbscan
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_2d = reducer.fit_transform(embeddings)
clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
clusterer.fit(umap_2d)

# 3 sorgu hasta
for i in [0, 100, 200]:
    result = run_pipeline(
        query_embedding=embeddings[i],
        reference_embeddings=embeddings,
        reference_labels=labels,
        umap_reducer=reducer,
        hdbscan_model=clusterer,
        patient_id=f"mock_{i:03d}",
    )
    print(json.dumps(result, indent=2))

# JSON dosyasına yaz
import json
with open("../outputs/pipeline_demo.json", "w") as f:
    json.dump([result1, result2, result3], f, indent=2)
```

---

### 13:00–14:00 | Öğle Yemeği + Mola

---

### 14:00–17:00 | ETHOS Bitir + JMIR Aging Başlat (3 saat)

**14:00–15:30 | ETHOS 3. Geçiş — Methods Bölümü**

Odak: Zero-shot yaklaşım nasıl implement ediliyor?

Not alınacaklar:
- Prompt template'i nasıl tasarlanmış?
- Zero-shot vs few-shot kararı nasıl verilmiş?
- Evaluation metriği ne? (Projenle karşılaştır: bizde C-Index / AUROC)
- Kendi projenle paralel: SleepFM embedding'den "zero-shot" hastalık tahmini —
  benzerlik var mı? Hangi noktada ayrılıyor?
- Bir paragraf yaz: ETHOS yaklaşımı ile senin similarity engine yaklaşımının farkı

**15:30–17:00 | JMIR Aging 1. Geçiş — Abstract + Figürler**

Odak: LLM prompt tasarımları

Not alınacaklar:
- Abstract'taki ana iddia tek cümleyle ne?
- Hangi figürler var, her biri ne gösteriyor?
- Heatmap veya matris görseli var mı? Kendi Modality Importance Matrix ile karşılaştır.
- Prompt tasarım prensipleri — en ilginç 2–3 tanesini not al.

---

### 17:00–17:30 | Mola

---

### 17:30–19:30 | visualization.py (2 saat)

**src/visualization.py** — iki temel fonksiyon. Seaborn + matplotlib kullanılacak.
(Plotly interaktif görseller ileriki günlere bırakılıyor — şu an static PNG yeterli.)

**Fonksiyon 1: `plot_umap_scatter`**
```python
def plot_umap_scatter(
    umap_coords: np.ndarray,           # (n_samples, 2)
    labels: np.ndarray,                # (n_samples,)
    disease_names: Dict[int, str],     # label → hastalık adı
    title: str = "UMAP Projection",
    highlight_idx: Optional[int] = None,  # query hastayı yıldızla işaretle
    save_path: Optional[str] = None,
) -> None:
    """Seaborn scatter plot. 12 hastalık grubu farklı renkte.

    highlight_idx verilirse o nokta büyük yıldız (★) ile işaretlenir —
    pipeline demo'da "bu hasta nerede?" sorusuna görsel cevap verir.
    """
```

**Fonksiyon 2: `plot_modality_importance_heatmap`**
```python
def plot_modality_importance_heatmap(
    weight_matrix: np.ndarray,   # (n_diseases, n_modalities)
    disease_names: List[str],    # satır etiketleri
    modality_names: List[str],   # sütun etiketleri: ["EEG", "ECG", "Resp", "EMG"]
    title: str = "Modality Importance Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Seaborn heatmap. Satır=hastalık, sütun=modalite.
    annotate=True (sayılar hücrelerde görünür).
    cmap="YlOrRd" — sıfır açık sarı, yüksek değer koyu kırmızı.

    Bu görsel SleepFM ekibine gönderilecek. Temiz, açıklamalı olmalı.
    """
```

**Mock Modality Importance Matrix üretimi ve kaydı:**
```python
# src içinde veya notebooks/02_pipeline_demo.ipynb'de:
from mock_data import build_disease_weight_profiles
from config import DISEASES, MODALITY_FULL_NAMES

profiles = build_disease_weight_profiles()
# profiles: Dict[str, np.ndarray] — disease_name → (128,) weight vector

# 12×4 matrix oluştur: her hastalık için modalite boyutlarının ortalama abs ağırlığı
weight_matrix = np.zeros((12, 4))
disease_list = [d.name_en for d in DISEASES.values()]
modality_list = ["eeg", "ecg", "resp", "emg"]

for i, disease_name in enumerate(disease_list):
    for j, mod in enumerate(modality_list):
        start, end = MODALITY_DIM_RANGES[mod]
        weight_matrix[i, j] = np.abs(profiles[disease_name][start:end]).mean()

plot_modality_importance_heatmap(
    weight_matrix=weight_matrix,
    disease_names=disease_list,
    modality_names=["EEG", "ECG", "Resp", "EMG"],
    title="Mock Modality Importance Matrix (12 Diseases × 4 Modalities)",
    save_path="outputs/mock_modality_importance.png",
)
```

**Beklenen görsel:** Kardiyak hastalıklar (Heart Failure, AFib) ECG sütununda
kırmızı; nörolojik hastalıklar (Dementia, Stroke, Depression) EEG sütununda kırmızı.

---

### 19:30–20:30 | Akşam Yemeği

---

### 20:30–22:00 | GitHub Commit + Entegrasyon Kontrolü (1.5 saat)

1. Tüm yeni dosyaları stage et: similarity_engine.py, pipeline.py, visualization.py,
   src/__init__.py, tests/__init__.py, notebooks/, outputs/
2. Test suite'i son kez çalıştır: `python -m pytest tests/ -v`
   Sonuç: en az 20 test geçmeli
3. Pipeline notebook'ta uçtan uca test et (run_pipeline 3 hasta)
4. JSON çıktısının geçerli olduğunu doğrula: `json.loads(json.dumps(result))` hata vermemeli
5. Commit mesajı: `"Day 6: folder structure, similarity engine, pipeline, visualization"`
6. 15 dk gün sonu notu yaz (ne öğrendim, ne zorlandım, gün 7 için dikkat noktası)

---

### 22:00–01:00 | Tampon + Tamamlama (3 saat)

Yetişmeyen şeyler için tampon. Muhtemel adaylar:
- Eksik testler: tests/test_similarity_engine.py (en az 5 test)
- JSON şema validasyonu (jsonschema kütüphanesi ile)
- UMAP görselinin daha iyi annotasyonu
- Gün 7 için ön hazırlık notu: LOO-CL eğitim döngüsünü nerede arayacaksın?
  (`pretrain.py` içinde `InfoNCE` veya `contrastive` kelimesini ara)

---

## GÜN SONU KONTROL LİSTESİ
- [ ] Klasör yapısı kuruldu: src/__init__.py, tests/__init__.py, notebooks/, outputs/
- [ ] Dosyalar taşındı, conftest.py güncellendi
- [ ] `python -m pytest tests/ -v` → 20 test geçiyor
- [ ] Mock data UMAP scatter plot üretildi (outputs/mock_umap.png)
- [ ] similarity_engine.py yazıldı: build_reference_index() + query_top5()
- [ ] pipeline.py yazıldı: run_pipeline() → JSON-serializable dict
- [ ] 3 mock hasta için pipeline çalıştırıldı → outputs/pipeline_demo.json üretildi
- [ ] visualization.py yazıldı: plot_umap_scatter() + plot_modality_importance_heatmap()
- [ ] Mock Modality Importance Matrix heatmap üretildi (outputs/mock_modality_importance.png)
- [ ] ETHOS 3. geçiş tamamlandı, notlar alındı
- [ ] JMIR Aging 1. geçiş tamamlandı, notlar alındı
- [ ] GitHub commit atıldı
- [ ] 15 dk gün sonu notu yazıldı

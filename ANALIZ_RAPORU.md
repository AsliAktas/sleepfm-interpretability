# SleepFM Interpretability — Kapsamlı Analiz Raporu

> ⚠️ **Bu rapor Nisan 2026 mock evrenine aittir.** İçindeki bug'ların
> bir kısmı mock katmanda hâlâ mevcut ama bu katman **legacy** olarak
> işaretlenmiştir ve bilimsel çıkarım için kullanılmıyor. Gerçek MESA
> cohort analizi için: [reports/phase15_defensibility/README.md](reports/phase15_defensibility/README.md).
> Güncel mimari: [README.md](README.md).

> **Tarih:** 17 Nisan 2026  
> **Yöntem:** Statik kod okuma + canlı test çalıştırma (8 yaratıcı test)  
> **Kural:** Kodda hiçbir değişiklik yapılmadı — saf gözlem raporu  
> **Dil:** Başlıklar TR, teknik detaylar EN

---

## ÖZET PUAN KARTI

| Katman | Durum | Not |
|---|---|---|
| Pipeline çalışıyor mu? | ✅ Evet | `exit code 0`, JSON üretiliyor |
| 20 test geçiyor mu? | ✅ Evet | Ama test edilmeyen kritik katmanlar var |
| Çıktılar anlamlı mı? | ⚠️ Kısmen | 1 hastalık tamamen kayıp, 1 hastalık hiç görünmüyor |
| Interpretability amacına uygun mu? | ❌ Hayır | Modality scores klinik olarak yorumlanamıyor |
| Güvenlik | ✅ Sorun yok | Offline proje, OWASP riski yok |
| Sürdürülebilirlik | ⚠️ Orta | Kritik modüller test edilmemiş |

---

## HATALAR — ÖNEMSİRALAMASI

---

### 🔴 HATA #1 — Hastalık İsimleri Kaydı (Label Index Mismatch)
**Önem: KRİTİK** | Dosyalar: `src/mock_data.py` ↔ `src/config.py` ↔ `src/similarity_engine.py`

#### 1A — Teknik Olmayan Dil

Hayal edin 12 çekmeceli bir dolabınız var ve çekmecelere 1'den 12'ye kadar numara yapıştırdınız. Ama içine koyduğunuz kağıtlar 0'dan 11'e kadar numaralandırılmış. Sonuç: 1. çekmeceye bakıyorsunuz, orada 0 numaralı bir kağıt var ama o numarayı tanımıyorsunuz çünkü etiketiniz 1'den başlıyor. 12. çekmece ise tamamen boş — Anksiyete Bozuklukları hastası asla oraya konmamış.

**Kullanıcı etkisi:** Sistem, 500 hastanın 40'ı için (%8) hiçbir hastalık adı söyleyemiyor. Anksiyete Bozuklukları ise tüm sistemden tamamen yok, hiçbir hastaya hiçbir zaman önerilmiyor.

#### 1B — Teknik + Genel Dil

`mock_data.py` şunu üretiyor:
```
labels = np.arange(n_disease_groups)   →  [0, 1, 2, ..., 11]
```
`config.py` şunu bekliyor:
```python
DISEASE_NAMES = {1: "Heart Failure", 2: "AFib", ..., 12: "Anxiety Disorders"}
# Anahtarlar: 1–12
```
`similarity_engine.py` içindeki `query_top5()` ise `DISEASE_NAMES.get(label)` çağırıyor.

**Test sonucu (canlı doğrulandı):**
```
label=0 olan hasta sayısı : 40 / 500
DISEASE_NAMES.get(0)       : MISSING  →  "Unknown(0)" döner
label=12 olan hasta sayısı : 0
DISEASE_NAMES.get(12)      : "Anxiety Disorders"  →  hiç üretilmedi
```

**Etki:** Her 500 hastadan 40'ının en yakın komşuları arasında label=0 içeriyorsa,
o komşuluk sonucu `"Unknown(0)"` olarak çıkar. 12. hastalık olan "Anxiety Disorders"
veri kümesinde hiç mevcut değil; dolayısıyla **tüm çalıştırmalarda sıfır kez önerilir.**
##### ÇÖZÜLEBİLİR BİR HATA
---

### 🔴 HATA #2 — `modality_scores` Klinik Olarak Yorumlanamıyor
**Önem: KRİTİK (Proje Amacı İçin)** | Dosya: `src/pipeline.py`

#### 2A — Teknik Olmayan Dil

Proje şunu sormayı hedefliyor: "Bu hastanın tahmini için beyin sinyalleri mi, kalp sinyalleri mi daha belirleyici oldu?" Ama şu an sistem o soruyu **yanıtlamıyor.** Bunun yerine "beyin bölgesindeki sayıların ortalaması nedir?" sorusunu yanıtlıyor. Bu iki soru tamamen farklı şeyler. Birinci soru yorumlanabilirlik (interpretability), ikincisi ise ham istatistik.

**Somut örnek:** İnme (Stroke) nörolojik bir hastalık — beyin sinyalleri (EEG) dominant beklenir. Ama sistem EKG'yi dominant gösteriyor:
```
mock_200 (Stroke): eeg=0.035, ecg=0.1192, resp=0.0641, emg=0.0237
                   dominant=ECG  ← klinik olarak yanlış
```

#### 2B — Teknik + Genel Dil

Mevcut kod:
```python
modality_scores[modality] = np.abs(query_embedding[start:end]).mean()
```
Bu satır, L2-normalize edilmiş embedding'in alt-vektörünün ortalama mutlak değerini hesaplıyor.

**Neden yanıltıcı?**
- L2 normalizasyon global: `||embedding||₂ = 1.0` her hastada. Bu, tüm 128 boyutun
  karelerinin toplamının 1'e eşit olduğu anlamına gelir. Bir boyut grubunun (ör. ECG)
  yüksek çıkması, o grubun "prediksiyon için önemli olduğu" anlamına **gelmez** —
  sadece o boyutlarda random büyüklük olduğu anlamına gelir.
- Gerçek interpretability için gereken: **attribution** — "ECG boyutlarını sıfırlasaydım
  benzerlik skoru ne kadar değişirdi?" (ablation) ya da gradient/SHAP tabanlı yöntem.

**Test sonucu — label=1 (ECG-dominant cluster) için modality scores:**

| Hasta | EEG | ECG | Resp | EMG | Dominant | Beklenen |
|---|---|---|---|---|---|---|
| Çoğu label=1 hastası | ~0.04 | ~0.09 | ~0.05 | ~0.04 | ECG | ECG ✅ |

Şans eseri label=1 hastalarında ECG biraz öne çıkıyor — ama bu tasarımdan değil,
normalizasyon sonrası rastgele kalmış büyüklükten kaynaklanıyor. Farklı hastalarda
bu tutarlılık bozuluyor (Stroke örneği yukarıda).
##### ÇÖZÜM ARAŞTIR, SÖZ KONUSU PLANIMIZDA ÇÖZÜM VAR MI? ATTENTION ALGORITHM? CLASSIFICATION ALGORITHM? ATTRIBUTION ZATEN ANA PROJEDE YAPILMIYOR MU? ÖZGÜNLÜK NEREDE?
---

### 🟠 HATA #3 — KNN Her Çağrıda Sıfırdan Kuruluyor
**Önem: ÖNEMLİ (Ölçeklenmez)** | Dosya: `src/pipeline.py`

#### 3A — Teknik Olmayan Dil

Bir kütüphanede 500 kitap var ve bir okuyucu için "en yakın konudaki 10 kitabı bul" diyoruz. Sistem her okuyucu için tüm 500 kitabı yeniden kataloglayıp sonra arıyor. 3 okuyucu için 3 kez katalog kuruldu. 500 okuyucu için 500 kez kurulur. Bu hem gereksiz hem de yavaş.

#### 3B — Teknik + Genel Dil

```python
def run_pipeline(query_embedding, reference_embeddings, ...):
    index = build_reference_index(reference_embeddings, n_neighbors=n_neighbors)  # ← her çağrıda fit()
    top5 = query_top5(query_embedding, index, ...)
```

`build_reference_index` içinde `NearestNeighbors.fit(embeddings)` çağrısı var.
500 hasta × 500 referans matrisinde bu fit işlemi her seferinde `O(n·d)` maliyet taşır.
n=10.000 olduğunda (gerçek veri) her `run_pipeline` çağrısı saniyeler sürer.

**Doğru tasarım:** Index dışarıda bir kez kurulur, `run_pipeline`'a parametre olarak geçirilir.
##### ÇÖZÜLEBİLİR BİR HATA
---

### 🟠 HATA #4 — KNN Sorgusu Hastanın Kendisini Buluyor (Self-Reference)
**Önem: ÖNEMLİ** | Dosyalar: `src/similarity_engine.py`, `test_pipeline_quick.py`

#### 4A — Teknik Olmayan Dil

Bir hastayla "sana en çok benzeyen hastalar kim?" diye soruyoruz. Ama o hasta zaten referans veritabanında da var. Sistem "en çok sana benzeyen kişi sensin" diyor ve onu da listeye ekliyor. Bu şu anlama gelir: gerçek 10 komşudan sadece 9 yeni hasta görüyoruz, 1 slot boşa gidiyor. Üstelik hastayla kendisi arasındaki mesafe sıfır olduğundan benzerlik skoru yapay olarak yüksek çıkıyor.

#### 4B — Teknik + Genel Dil

`test_pipeline_quick.py` ve `02_pipeline_demo.ipynb` içinde:
```python
r = run_pipeline(
    query_embedding=embeddings[i],      # hasta i
    reference_embeddings=embeddings,    # hasta i de referansta var
)
```

**Test sonucu (canlı doğrulandı):**
```
Patient 0 → kneighbors() çağrısı
1. komşu mesafesi: 0.0000  →  is_self: True
```

`sklearn.NearestNeighbors` default olarak self-exclusion yapmaz.
Etki:
- `similarity = 1 - 0/2 = 1.0` → yapay en yüksek skor
- Oy sayımında kendi label'ı 1 ekstra oy alıyor → `n_votes` şişiyor
- `mock_000` için `n_votes=7` — 10 komşudan 7'si Obesity. Bu kısmen self-reference etkisi.
##### MUHTEMELEN ÇÖZÜLEBİLİR BİR HATA
---

### 🟡 HATA #5 — Perturbation Gerçek Komorbiditesi Simüle Etmiyor
**Önem: TASARIM KAYGISI** | Dosya: `src/mock_data.py`

#### 5A — Teknik Olmayan Dil

Gerçek dünyada hem kalp yetmezliği hem diyabeti olan bir hastanın vücut sinyalleri (embedding) her iki hastalığın izlerini taşır. Ama bu sistemde "karışık hasta" sadece etiketi değiştirilerek simüle ediliyor — sanki kalp yetmezliği sinyali üreten ama diyabet etiketi taşıyan bir hasta. Bu gerçekçi değil, sadece "yanlış etiketlenmiş" hastayı simüle ediyor.

#### 5B — Teknik + Genel Dil

```python
# Step 3: Perturbation — reassign labels, embeddings stay at original cluster
final_labels[perturb_idxs] = (final_labels[perturb_idxs] + offsets) % n_disease_groups
# Embedding değişmiyor!
```

**Test sonucu (T8 - yaratıcı test):**
- 75 hasta perturbe edildi (label değişti, embedding değişmedi)
- Bu hastaları sorguladığında KNN, embedding'e bakarak **orijinal** label'ı tahmin ediyor
- Perturbe edilmiş hastalar için tahmin ≠ atanmış label → pipeline çıktısı "yanlış" görünüyor

**Klinik sonuç:** "15% misclassification oranı" iddiası doğru değil — embedding ve
label uyuşmazlığından dolayı yapay bir karışıklık var, gerçek komorbidite değil.
##### ÇÖZÜLEBİLİR BİR HATA
---

### 🟡 HATA #6 — Test Coverage Kritik Modülleri Kapsıyor
**Önem: SÜRDÜRÜLEBİLİRLİK** | Dosya: `tests/`

#### 6A — Teknik Olmayan Dil

Bir araba fabrikasını hayal edin. Sadece motor testleri yapılıyor. Frenleri, direksiyonu, lambaları hiç test edilmiyor. Motor mükemmel çalışıyor olabilir ama araç güvenli değil.

#### 6B — Teknik + Genel Dil

| Modül | Test Durumu | Risk |
|---|---|---|
| `mock_data.py` | ✅ 20 test | Tam kapsam |
| `config.py` | ✅ Dolaylı | Yeterli |
| `similarity_engine.py` | ❌ **SIFIR TEST** | Label Bug burada gizleniyor |
| `pipeline.py` | ❌ **SIFIR TEST** | Self-reference, KNN rebuild burada |
| `metrics.py` | ❌ **SIFIR TEST** | AUROC, C-Index, bootstrap doğrulanmadı |
| `visualization.py` | ❌ **SIFIR TEST** | Çıktı kalitesi kontrol yok |

**Kritik gözlem:** Hata #1 (label mismatch) tam olarak test edilmeyen `similarity_engine.py`
içinde ortaya çıkıyor. 20 test %100 geçiyor ama sistemin asıl gözlemlenebilir çıktısını
üreten kod hiç test edilmemiş.
##### ÇÖZÜLEBİLİR HATA : TEST YAZMADAN KOD YAZILMAYACAK (YENİ KODLAR İÇİN), HER KODUN TESTİ OLACAK (MEVCUT KODLAR VE GELECEKTEKİ KODLARI KAPSAR).
---

### 🟢 HATA #7 — Similarity Formülü Dokümantasyon Eksikliği
**Önem: DÜŞÜK** | Dosya: `src/similarity_engine.py`

#### 7A — Teknik Olmayan Dil

Kullandığınız benzerlik ölçüsü "cosine similarity" sanılabilir ama değil. Matematiksel olarak doğru bir formül ama isminin ne olduğu ve neden bu formülün seçildiği hiçbir yerde açıklanmıyor.

#### 7B — Teknik + Genel Dil

Formül: `similarity = 1 - (mean_distance / 2)`

L2-normalize vektörler için Öklid mesafesi $d \in [0, 2]$ aralığındadır. Bu formül $d$'yi $[0,1]$'e lineer olarak map eder.

Cosine similarity şu şekilde hesaplanır: $\cos\theta = 1 - d^2/2$ (farklı)

**Test sonucu (T7 - yaratıcı test):**
```
50 hasta × 10 komşu = 500 mesafe değeri
Max mesafe: 1.xxxx (< 2.0)  →  Negatif similarity: Hayır
Formül matematiksel olarak geçerli.
```

Sorun yok — sadece "cosine similarity" ile karıştırılmaması için açıklama eklenmeli.

---

## ÇIKTI ANALİZİ — `pipeline_demo.json` ve `outputs/`

### JSON Çıktısı Detaylı İncelemesi

#### `mock_000` — Obesity (Label: 8)

```json
{
  "embedding_norm": 1.0,        ✅ L2-normalize çalışıyor
  "cluster_id": 0,              ✅ HDBSCAN atadı
  "top5_diseases": [
    {"rank": 1, "disease": "Obesity", "similarity": 0.7838, "n_votes": 7}
  ],
  "modality_scores": {"eeg": 0.0257, "ecg": 0.1168, "resp": 0.0634, "emg": 0.0348}
}
```

| Alan | Değerlendirme |
|---|---|
| `embedding_norm = 1.0` | ✅ Doğru |
| `cluster_id = 0` | ✅ HDBSCAN 12 küme buldu, mantıklı |
| Top-1: Obesity | ✅ Bu hasta muhtemelen Obesity cluster'ından geliyor |
| `n_votes = 7` | ⚠️ Self-reference nedeniyle şişmiş (1 slot kendi) |
| ECG en yüksek modality score | ❌ Obesity için ECG dominant beklenemez |
| 4 sonuç döndü (top-5 değil) | ⚠️ 10 komşu arasında yalnızca 4 farklı hastalık var |

#### `mock_100` — Depression (Label: 5)

```json
"top5_diseases": [
  {"disease": "Depression", "similarity": 0.7926, "n_votes": 6},
  {"disease": "Chronic Kidney Disease", "n_votes": 2},
  {"disease": "Hypertension", "n_votes": 1},
  {"disease": "Heart Failure", "n_votes": 1}
],
"modality_scores": {"eeg": 0.0323, "ecg": 0.0295, "resp": 0.1056, "emg": 0.0608}
```

| Alan | Değerlendirme |
|---|---|
| Top-1: Depression | ✅ Tutarlı |
| Resp en yüksek modality | ⚠️ Depresyon için Resp dominant? Beklenti EEG dominant |
| n_votes toplamı = 10 | ✅ n_neighbors=10 ile tutarlı |

#### `mock_200` — Stroke (Label: 4)

```json
"top5_diseases": [
  {"disease": "Stroke", "similarity": 0.7744, "n_votes": 8}
],
"modality_scores": {"eeg": 0.035, "ecg": 0.1192, "resp": 0.0641, "emg": 0.0237}
```

| Alan | Değerlendirme |
|---|---|
| Top-1: Stroke | ✅ Tutarlı |
| ECG en yüksek modality | ❌ **İnme nörolojik → EEG dominant beklenir, ECG değil** |
| n_votes = 8 | ⚠️ 10 komşunun 8'i Stroke — küme çok sıkı = UMAP aşırı ayrışmış olabilir |

### UMAP Görsel Analizi

Görselden gözlemler:

| Gözlem | Değerlendirme |
|---|---|
| 12 küme net ayrışmış | ⚠️ Aşırı iyi ayrışma — gerçek veri bu kadar temiz olmaz |
| Kümeler arası boşluklar büyük | ⚠️ `perturbation_rate=0.15` etkisi görsel olarak yok denecek kadar az |
| Query hasta (★) Stroke yakınında | ✅ Tutarlı |
| Bazı kümeler çok küçük nokta kümesi | ⚠️ ~40 hasta bir araya sıkışmış → UMAP over-compresses small clusters |

### Heatmap Analizi (Modality Importance Matrix)

Heatmap `mock_data.py`'deki tasarım ağırlıklarını (`_DOMINANT=2.0`, `_SECONDARY=1.0`, `_BACKGROUND=0.3`) doğrudan görselleştiriyor — gerçek veriden hesaplanmıyor.

| Gözlem | Değerlendirme |
|---|---|
| Her satırda 1 dominant (2.0) var | ✅ Tasarıma uygun |
| Değerler 0.300 / 1.000 / 2.000 | ⚠️ Ham weight'ler — gerçek hasta verisiyle değişecek |
| Kalp Yetmezliği → ECG dominant | ✅ Klinik olarak doğru tasarım |
| Depresyon → EEG dominant (2.0) | ✅ Klinik olarak doğru tasarım |
| **Ama pipeline çıktısı bunu yansıtmıyor** | ❌ Heatmap ile modality_scores çelişiyor |

---

## INTERPRETABILITY AMACINA UYUMLULUK ANALİZİ

Projenin temel sorusu: *"Bu hastanın tahmini için hangi sinyal (EEG/ECG/Resp/EMG) belirleyiciydi?"*

### Mevcut Durum

```
Patient → embedding → KNN → Top-5 hastalık  ✅ (kısmen çalışıyor)
                   → modality_scores         ❌ (soruyu yanıtlamıyor)
```

### Interpretability Puanlama

| Interpretability Bileşeni | Mevcut mu? | Doğru mu? |
|---|---|---|
| "Hasta hangi hastalığa benziyor?" | ✅ var | ⚠️ label bug nedeniyle eksik |
| "Ne kadar benziyor?" (similarity) | ✅ var | ✅ matematiksel olarak doğru |
| "Hangi sinyal belirleyici?" | ⚠️ var ama... | ❌ yanlış hesaplanıyor |
| "Embedding uzayında nerede?" (UMAP) | ✅ var | ✅ doğru |
| "Hangi kümedeyim?" (HDBSCAN) | ✅ var | ✅ doğru |
| "Risk skorum nedir?" (C-Index) | ⚠️ kod var | ❌ hiç test edilmemiş |
| "Komorbidite var mı?" | ❌ yok | — |

**Kritik boşluk:** Projenin başlığı "Interpretability" ama tek gerçek interpretability
bileşeni olan "hangi modality önemli?" sorusu şu an yanlış hesaplanıyor. Geri kalan her şey
(KNN, UMAP, HDBSCAN) **kümeleme ve benzerlik** — interpretability değil.

---

## NEYİ KALDIR, NEYİ EKSİLT, NEYİ BIRAK

### ✅ KALACAKLAR (Dokunma)

- `config.py` tamamı — mükemmel tasarım, validasyon, tek kaynak
- `mock_data.py` tamamı — sağlam, test edilmiş, iyi belgelenmiş
- `metrics.py` AUROC implementasyonu — tie-handling ile doğru
- `utils.py` tamamı — `normalize_l2`, `set_all_seeds` sağlam
- `conftest.py` — pytest path yönetimi temiz
- UMAP + HDBSCAN pipeline'ın kendisi — doğru ve çalışıyor

### ❌ DEĞİŞMESİ GEREKENLER

| Ne | Nerede | Nasıl |
|---|---|---|
| `np.arange(0, 12)` → `np.arange(1, 13)` | `mock_data.py` | Label/name hizalaması |
| `modality_scores` hesabı | `pipeline.py` | Ablation veya attribution ile değiştirilmeli |
| KNN index pipeline içinde kurulmasın | `pipeline.py` | Dışarıda kurulup parametre olarak geçirilmeli |
| `test_pipeline_quick.py` kök dizinde | proje yapısı | `tests/` klasörüne taşınmalı |

### ➕ EKLENMESİ GEREKENLER

| Ne | Neden |
|---|---|
| `tests/test_similarity_engine.py` | "Unknown()" çıktısını yakalayacak test |
| `tests/test_pipeline.py` | Self-reference, çıktı formatı, json serializability |
| `tests/test_metrics.py` | AUROC, C-Index edge case'leri |
| Gerçek attribution metodu | Projenin interpretability amacı için zorunlu |

### ➖ EKSİLTİLMESİ GEREKENLER

| Ne | Neden |
|---|---|
| `requirements.txt` sabit versiyonlar | Ortamla uyumsuz (numpy 1.24.4 ama ortamda 2.2.6) |
| `pretrain.py` proje kökünde | SleepFM orijinal kodu — bu projeyle ilgisi yok, karışıklık yaratıyor |

---

## YARATICI TEST SONUÇLARI — ÖZET

| Test # | Ne Test Edildi | Sonuç | Bulgu |
|---|---|---|---|
| T1 | Label=0 ve label=12 existence | **Doğrulandı** | 40 hasta "Unknown(0)", Anxiety Disorders hiç yok |
| T2 | `query_top5` gerçek çıktısı | **Doğrulandı** | Patient 0: 4 sonuç, hepsi mevcut hastalıklar |
| T3 | Self-reference (mesafe=0) | **Doğrulandı** | `distance[0][0] = 0.0`, `is_self = True` |
| T4 | Anxiety Disorders görünürlüğü | **Doğrulandı** | Tüm output'larda 0 kez |
| T5 | 500 hasta üzerinde Unknown sayımı | 🔄 Çalışıyor | label=0 içeren komşulukta Unknown() çıkar |
| T6 | ECG-dominant cluster → modality_scores | **Doğrulandı** | Çoğu hastada ECG öne çıkıyor (rastlantısal) |
| T7 | Similarity formülü aralık kontrolü | **Geçti** | Negatif similarity yok, formül matematiksel geçerli |
| T8 | Perturbation → embedding/label bağlantısı | **Doğrulandı** | KNN perturbe hastayı orijinal cluster'a atıyor |

---

## SONUÇ

Bu proje, SleepFM interpretability analizi için **sağlam bir iskelet** üzerine kurulmuş.
Veri üretimi, kümeleme (UMAP + HDBSCAN) ve benzerlik motoru (KNN) teknik olarak işlevsel.

Ancak projenin **öz amacı olan interpretability** — "bu tahmini hangi sinyal sürükledi?" sorusu — şu an yanlış bir metrikle yanıtlanıyor. Bu, pipeline çıktılarına bakıldığında görsel
olarak fark edilmiyor çünkü sayılar makul görünüyor; ama klinik bağlam konulduğunda
(İnme → ECG dominant?) çelişki ortaya çıkıyor.

**En acil düzeltme:** Label index mismatch (#1) — tek satır değişiklik, somut ve ölçülebilir etkisi var.  
**En önemli geliştirme:** Gerçek attribution metodu (#2) — projenin var olma sebebi.

```
Çalışan Şeyler:   pipeline ✅ | UMAP ✅ | HDBSCAN ✅ | KNN ✅ | normalizasyon ✅
Sessiz Hatalar:   label mismatch ❌ | self-reference ❌ | modality_scores semantiği ❌
Test Boşlukları:  similarity_engine ❌ | pipeline ❌ | metrics ❌
Interpretability: %30 (benzerlik var, attribution yok)
```

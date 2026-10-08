# GÜN 7 — Hafta 1 Değerlendirme + LOO-CL Anlayışı + Yazı

## BUGÜNÜN HEDEFİ
Gün sonunda elinde şunlar olacak:
1. `tests/test_similarity_engine.py` yazılmış: 8 test, hepsi geçiyor
2. `tests/test_pipeline.py` yazılmış: 8 test, hepsi geçiyor → toplam ≥ 36 test
3. LOO-CL mekanizması `pretrain.py` üzerinden satır satır anlaşılmış
4. InfoNCE formülü kağıda yazılmış; ablation gerekçesi 1 paragraf olarak tamamlanmış
5. `outputs/HAFTA1_OZET.md` taslağı hazır (commit edilmeyecek)
6. Anlamadığın 5 kavram listesi oluşturulmuş
7. JMIR Aging 2. geçiş + JAMIA EHR Phenotyping 1. geçiş tamamlanmış
8. Hafta 1 kontrolü yapılmış, GitHub commit atılmış

---

## GÜN 6'DAN DEVİR ALINAN GÖREVLER

| Görev | Durum | Gün 7'deki yeri |
|---|---|---|
| `test_similarity_engine.py` | ❌ Buffer'a kalmıştı | 10:00–11:00 |
| `test_pipeline.py` | ❌ Buffer'a kalmıştı | 11:00–11:30 |
| `PROJECT_OVERVIEW.md` commit | ⏳ Untracked | 19:00 commit bloğunda |

---

## GÜN 6 HIZLI HATIRLATMA

### Gün 6 Çıktıları
- **similarity_engine.py**: `build_reference_index()` (KNN, k=10, euclidean),
  `query_top5()` (label voting, similarity = 1 - mean_dist/2)
- **pipeline.py**: `run_pipeline()` → JSON-serializable dict
  (patient_id, timestamp, embedding_norm, umap_coords, cluster_id, top5_diseases, modality_scores)
- **visualization.py**: `plot_umap_scatter()`, `plot_modality_importance_heatmap()`
- **notebooks/02_pipeline_demo.ipynb**: 3 mock hasta için uçtan uca pipeline
- **outputs/pipeline_demo.json**: Gerçek pipeline çıktısı mevcut
- **20 test**: `tests/test_embedding_generation.py` — hepsi geçiyor

### Önemli Hatırlatmalar
- `conftest.py` root'ta, `sys.path`'e `src/` eklenmiş
- Embedding: 128 boyut, L2-normalize, 4×32: eeg(0:32), ecg(32:64), resp(64:96), emg(96:128)
- `pretrain.py` proje root'unda mevcut (orijinal SleepFM dosyası)

---

## SAAT SAAT PROGRAM

### 10:00–11:30 | Test Dosyaları (1.5 saat) [Gün 6'dan devir]

**Neden önce testler:**
LOO-CL okumadan önce Day 6 borcunu kapat. Test yazma ısınma görevi görür ve
similarity_engine + pipeline'ı tekrar aklında canlandırır.

---

#### 10:00–11:00 | `tests/test_similarity_engine.py`

**Yapı:**
```python
import pytest
import numpy as np
from similarity_engine import build_reference_index, query_top5
from config import DISEASE_NAMES


@pytest.fixture
def mock_index_data():
    np.random.seed(42)
    n = 100
    embeddings = np.random.randn(n, 128)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    labels = np.tile(np.arange(12), n // 12 + 1)[:n]
    return embeddings, labels


class TestBuildReferenceIndex:

    def test_returns_fitted_nn(self, mock_index_data):
        """build_reference_index() sklearn NearestNeighbors döndürmeli."""
        from sklearn.neighbors import NearestNeighbors
        embeddings, _ = mock_index_data
        index = build_reference_index(embeddings)
        assert isinstance(index, NearestNeighbors)

    def test_default_n_neighbors(self, mock_index_data):
        """Varsayılan n_neighbors=10 olmalı."""
        embeddings, _ = mock_index_data
        index = build_reference_index(embeddings)
        assert index.n_neighbors == 10

    def test_custom_n_neighbors(self, mock_index_data):
        """n_neighbors parametresi dışarıdan geçilebilmeli."""
        embeddings, _ = mock_index_data
        index = build_reference_index(embeddings, n_neighbors=5)
        assert index.n_neighbors == 5


class TestQueryTop5:

    def test_returns_list(self, mock_index_data):
        """query_top5() bir list döndürmeli."""
        embeddings, labels = mock_index_data
        index = build_reference_index(embeddings)
        result = query_top5(embeddings[0], index, labels, DISEASE_NAMES)
        assert isinstance(result, list)

    def test_returns_at_most_5(self, mock_index_data):
        """Sonuç en fazla 5 eleman içermeli."""
        embeddings, labels = mock_index_data
        index = build_reference_index(embeddings)
        result = query_top5(embeddings[0], index, labels, DISEASE_NAMES)
        assert len(result) <= 5

    def test_result_dict_has_required_keys(self, mock_index_data):
        """Her eleman rank, disease, similarity, n_votes anahtarlarını içermeli."""
        embeddings, labels = mock_index_data
        index = build_reference_index(embeddings)
        result = query_top5(embeddings[0], index, labels, DISEASE_NAMES)
        required_keys = {'rank', 'disease', 'similarity', 'n_votes'}
        for item in result:
            assert required_keys == set(item.keys())

    def test_similarity_in_valid_range(self, mock_index_data):
        """Her similarity değeri [0, 1] aralığında olmalı."""
        embeddings, labels = mock_index_data
        index = build_reference_index(embeddings)
        result = query_top5(embeddings[0], index, labels, DISEASE_NAMES)
        for item in result:
            assert 0.0 <= item['similarity'] <= 1.0

    def test_ranks_are_sequential(self, mock_index_data):
        """Rank'lar 1'den başlayıp art arda gelmeli."""
        embeddings, labels = mock_index_data
        index = build_reference_index(embeddings)
        result = query_top5(embeddings[0], index, labels, DISEASE_NAMES)
        ranks = [item['rank'] for item in result]
        assert ranks == list(range(1, len(result) + 1))

    def test_n_votes_sum_equals_k(self, mock_index_data):
        """Tüm oyların toplamı n_neighbors'a (k=10) eşit olmalı."""
        embeddings, labels = mock_index_data
        index = build_reference_index(embeddings, n_neighbors=10)
        result = query_top5(embeddings[0], index, labels, DISEASE_NAMES)
        assert sum(item['n_votes'] for item in result) == 10
```

**Çalıştır ve doğrula:**
```
python -m pytest tests/test_similarity_engine.py -v
```
Beklenen: **8 passed**

---

#### 11:00–11:30 | `tests/test_pipeline.py`

**Tasarım Notu — Neden `scope="module"` fixture:**
UMAP + HDBSCAN fit işlemi ~5-15 saniye sürer. `scope="module"` ile fixture tüm
test modülü boyunca bir kez çalıştırılır, testler ayrı ayrı her seferinde yeniden
fit etmez. Bu, 36+ testli suite'in hızını korur.

```python
import pytest
import json
import numpy as np

import umap
import hdbscan

from pipeline import run_pipeline
from mock_data import generate_mock_embeddings


@pytest.fixture(scope="module")
def pipeline_fixtures():
    """UMAP + HDBSCAN modülü boyunca bir kez fit edilir."""
    embeddings, labels = generate_mock_embeddings(seed=42, n_samples=100)
    reducer = umap.UMAP(n_components=2, n_neighbors=10, random_state=42)
    umap_2d = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    clusterer.fit(umap_2d)
    return embeddings, labels, reducer, clusterer


@pytest.fixture
def single_result(pipeline_fixtures):
    embeddings, labels, reducer, clusterer = pipeline_fixtures
    return run_pipeline(
        query_embedding=embeddings[0],
        reference_embeddings=embeddings,
        reference_labels=labels,
        umap_reducer=reducer,
        hdbscan_model=clusterer,
        patient_id="test_patient",
    )


class TestPipelineOutput:

    def test_returns_dict(self, single_result):
        assert isinstance(single_result, dict)

    def test_json_serializable(self, single_result):
        """run_pipeline() çıktısı JSON'a serileştirilebilmeli."""
        json.dumps(single_result)  # hata fırlatırsa test başarısız

    def test_required_keys_present(self, single_result):
        required = {
            'patient_id', 'timestamp', 'embedding_norm',
            'umap_coords', 'cluster_id', 'top5_diseases', 'modality_scores'
        }
        assert required.issubset(set(single_result.keys()))

    def test_patient_id_preserved(self, single_result):
        assert single_result['patient_id'] == "test_patient"

    def test_embedding_norm_approx_one(self, single_result):
        """L2-normalize edilmiş embedding normu ~1.0 olmalı."""
        assert abs(single_result['embedding_norm'] - 1.0) < 0.01

    def test_umap_coords_length(self, single_result):
        assert len(single_result['umap_coords']) == 2

    def test_modality_scores_keys(self, single_result):
        expected = {'eeg', 'ecg', 'resp', 'emg'}
        assert set(single_result['modality_scores'].keys()) == expected

    def test_top5_disease_schema(self, single_result):
        required_keys = {'rank', 'disease', 'similarity', 'n_votes'}
        for item in single_result['top5_diseases']:
            assert required_keys.issubset(set(item.keys()))
```

**Çalıştır — tüm suite:**
```
python -m pytest tests/ -v
```
Beklenen: **36 passed** (20 + 8 + 8)

---

### 11:30–13:00 | LOO-CL Mekanizması (1.5 saat)

#### 11:30–12:15 | `pretrain.py` Satır Satır Okuma

**`pretrain.py` dosyası proje root'unda. Açılacak bölümler:**

**Bölüm 1 — `run_iter()` fonksiyonu (satır 21–140)**

Tüm contrastive learning mantığı burada. Okurken şunları not al:

**Satır 21:** `def run_iter(batch, num_modalities, model, device, mode, temperature, batch_size, ij)`
→ `mode` parametresi: `"pairwise"` veya `"leave_one_out"`
→ `temperature`: Öğrenilebilir parametre (`torch.nn.parameter.Parameter`)

**Satır 50–65:** 4 modalite veri yüklemesi ve cihaza taşıma
```python
(bas, resp, ekg, emg) = batch_data
emb = [
    model(bas, mask_bas),
    model(resp, mask_resp),
    model(ekg, mask_ekg),
    model(emg, mask_emg),
]
```
→ `bas` = beyin aktivitesi (EEG/brainwave), `resp` = solunum, `ekg` = kalp, `emg` = kas
→ Her modalite bağımsız olarak model'den geçirilir, ayrı bir embedding üretilir.

**Satır ~66–70:** L2 normalizasyon
```python
for i in range(num_modalities):
    emb[i] = torch.nn.functional.normalize(emb[i])
```
→ **Bu bizim projeyle doğrudan bağlantılı:** config'deki `normalize_l2()` ile aynı mantık.

**Satır ~112 — `leave_one_out` modu:** Projenin asıl eğitim stratejisi burada başlıyor.
```python
if mode == "leave_one_out":
    for i in range(num_modalities):
        other_emb = torch.stack(
            [emb[j] for j in list(range(i)) + list(range(i + 1, num_modalities))]
        ).sum(0) / (num_modalities - 1)
```
→ **Kritik satır:** `i` numaralı modalite **hariç** diğerlerinin embedding ortalaması alınır.
→ Örnek: i=1 (resp) için → `other_emb = mean(bas, ekg, emg)` — resp yokmuş gibi davranılır.
→ Sonra `emb[i]` bu `other_emb` ile eşleştirilmeye çalışılır.

**Logits hesabı:**
```python
logits = torch.matmul(emb[i], other_emb.transpose(0, 1)) * torch.exp(temperature)
labels = torch.arange(logits.shape[0], device=device)
l = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
```
→ `logits[n, k]` = n. örneğin modalite-i embedding'i ile k. örneğin diğer-modalite ortalamasının
   kosinüs benzerliği (normalize vektörlerde matmul = cosine)
→ `labels` = diyagonal: her örnek yalnızca kendisiyle eşleşmeli
→ `cross_entropy` = InfoNCE kaybı

**Satır ~136:** `loss /= num_modalities * 2`
→ 4 modalite × 2 yön (i→-i ve -i→i) = 8 terim ortalaması

**Bölüm 2 — `pretrain()` fonksiyonu (satır ~143+)**

Kısaca gözden geçir:
- SGD optimizer + momentum (AdamW değil — neden? stabilite tercihi)
- `temperature` öğrenilebilir parametre olarak optimizer'a ekleniyor
- Her epoch: DataLoader → `run_iter()` → `loss.backward()` → `optim.step()`

---

#### 12:15–12:45 | InfoNCE Formülünü Yaz

**LOO-CL Formülü (kağıda yaz, notebooks/ veya gün sonu notuna da ekle):**

Tek bir modalite i için kayıp:

$$\mathcal{L}_{i \to -i} = -\frac{1}{N} \sum_{n=1}^{N} \log \frac{\exp\!\left(\mathbf{z}_i^{(n)} \cdot \bar{\mathbf{z}}_{-i}^{(n)} \,/\, \tau\right)}{\sum_{k=1}^{N} \exp\!\left(\mathbf{z}_i^{(n)} \cdot \bar{\mathbf{z}}_{-i}^{(k)} \,/\, \tau\right)}$$

Toplam LOO-CL kaybı:
$$\mathcal{L}_{LOO} = \frac{1}{2M} \sum_{i=1}^{M} \left(\mathcal{L}_{i \to -i} + \mathcal{L}_{-i \to i}\right)$$

**Değişkenler:**
- $M = 4$: modalite sayısı (bas, resp, ekg, emg)
- $N$: batch size
- $\mathbf{z}_i^{(n)}$: n. örneğin modalite-i embedding'i (L2-normalize)
- $\bar{\mathbf{z}}_{-i}^{(n)}$: n. örneğin i **hariç** diğer modalities ortalaması
- $\tau$: öğrenilebilir sıcaklık (`torch.exp(temperature)`)
- Payda: tüm batch içindeki N negatif (farklı örnekler) + 1 pozitif (kendisi)

**Bağlantı kuracak şey:** Bu formül sayesinde model, aynı hastanın farklı modalite
sinyallerinin birbirine yakın, farklı hastaların sinyallerinin uzak embedding üretmeyi öğrenir.
Bizim `similarity_engine.py`'deki euclidean KNN'in neden çalıştığının matematiksel temeli budur.

---

#### 12:45–13:00 | Ablation Gerekçesi (1 Paragraf)

Şu soruya cevap ver: **"Neden LOO-CL ile eğitilmiş modelde modalite ablasyonu geçerlidir?"**

Taslak (kendi kelimelerinle yeniden yaz — yaklaşık 100 kelime):

> SleepFM'in LOO-CL eğitimi, her mini-batch iterasyonunda kasıtlı olarak eksik modalite
> koşulunu simüle eder: modalite-i hariç tutularak diğerlerinin ortalamasıyla karşılaştırılır
> (`other_emb = mean(emb[j] for j ≠ i)`). Bu sayede model, eğitim sürecinde zaten
> "yokmuş gibi davranan" modaliteler görmüştür. Inference'ta bir modaliteyi sıfırlamak
> (ablasyon) distribüsyon kaymasına (OOD) yol açmaz; çünkü model eksik modaliteyle
> çalışmayı öğrenmiştir. Bu durum, ablasyon sonuçlarının güvenilir yorumlanmasını sağlar:
> bir modaliteyi kaldırdığımızda performans düşüşü, o modalitedeki gerçek sinyal katkısını
> yansıtır, model başarısızlığını değil.

Bu paragrafı `outputs/HAFTA1_OZET.md`'ye de yapıştır.

---

### 13:00–14:00 | Öğle Yemeği

---

### 14:00–15:00 | HAFTA1_OZET.md (1 saat)

**Dosya:** `outputs/HAFTA1_OZET.md` — **commit edilmeyecek**, kişisel referans.

**Yapı önerisi (her başlık ~2-4 cümle):**

```markdown
# Hafta 1 Özeti — SleepFM Interpretability

## SleepFM Ne Yapıyor?
...

## Ben Ne Ekliyorum?
...

## Temel Kavramlar

### CoxPH Nedir?
...

### Embedding Nedir? 128 Boyut Ne Anlama Geliyor?
...

### LOO-CL Nedir? (Bugün öğrenilen)
...

## Ablation Neden Geçerli?
[12:45-13:00 bloğunda yazdığın paragrafı buraya yapıştır]

## Hafta Sonu İtibarıyla Proje Durumu
...
```

**Yazarken zorlama:** Kağıda kalem gibi yaz — düzeltme yapmadan taslak at.
Amaç kayıt tutmak, yayın hazırlamak değil.

---

### 15:00–15:30 | Anlamadığım 5 Kavram (30 dk)

Bugüne kadar geçip "anladım" dediğin ama aslında net olmayan 5 kavramı listele.
Bu kavramları **Hafta 2'ye taşıma listesin** olacak.

**Yöntem:**
1. `HAFTA1_OZET.md`'ye `## Gelecek Haftaya Taşınan Kavramlar` başlığı ekle
2. Her kavram için: `- [ ] KavramAdı — neden belirsiz (1 cümle)`

**Örnek adaylar** (senden gelmeli ama bunlar tipik olanlar):
- SetTransformer ile standart Transformer arasındaki fark ne?
- PSG kaydındaki `bas` sinyali tam olarak ne ölçüyor?
- HDBSCAN'da `-1` cluster etiketi ne anlama geliyor (noise point)?
- CoxPH'da "partial likelihood" neden tam likelihood değil?
- Temperature parametresi neden öğrenilebilir yapılmış, sabit bırakılmayıp?

---

### 15:30–16:15 | JMIR Aging 2. Geçiş (45 dk)

**Odak: Methods bölümü — Prompt tasarımları + Değerlendirme metrikleri**

Gün 6'da Abstract + Figürleri okudun. Bu geçişte:

Not alınacaklar:
1. LLM prompt template'i tam olarak nasıl yapılandırılmış?
   - Few-shot mu, zero-shot mu, chain-of-thought mu?
   - Hangi hastalık bağlamları kullanılmış?
2. Başarı metriği ne? (AUC, accuracy, kappa?)
3. **En az 2 cümle yaz:** Bu çalışmanın değerlendirme yaklaşımıyla bizim
   `metrics.py`'deki `concordance_index` + `auroc` kombinasyonunun farkı nedir?
4. Figürlerden biri heatmap veya matris görseli içeriyor mu?
   → Varsa: `outputs/mock_modality_importance.png` ile tasarım karşılaştırması yap.

---

### 16:15–17:00 | JAMIA EHR Phenotyping 1. Geçiş (45 dk)

**Odak: Abstract + Figürler + Giriş son paragrafı**

Not alınacaklar:
1. Abstract'ın tek cümlelik iddiası ne? (kağıda yaz)
2. Kaç figür var, her biri ne gösteriyor? (figür başlıklarına bakarak listele)
3. EHR phenotyping ile SleepFM'deki "hastalık kümelenmesi" arasındaki kavramsal
   fark nedir? (1-2 cümle — kendi kelimelerinle)
4. Giriş son paragrafında gelecek çalışma olarak ne öneriyor?
   → Bu projeyle örtüşüyor mu?

---

### 17:00–17:30 | Mola

---

### 17:30–19:30 | Hafta Kontrolü (2 saat)

Bu blok 4 kontrol + 1 commit + 1 gün sonu notundan oluşuyor.

---

#### 17:30–18:15 | Kontrol 1 & 2

**Kontrol 1: SetTransformer forward() akışını anlatabilme**

`pretrain.py`'de modelin şöyle çağrıldığını gördün: `model(bas, mask_bas)`.
Şimdi şu soruyu **yüksek sesle veya yazarak** bir kişiye anlatırmış gibi yanıtla:

> "SetTransformer'a bir modaliteden gelen sinyal parçaları (patches) girdi olarak geliyor.
> İçeride ne oluyor? Nasıl bir embedding çıkıyor?"

Beklenen akış (bu kısımları sayabiliyorsan geçtin):
1. Sinyal → sabit boyutlu patch'lere böl (patch_size konfigürasyonda)
2. Her patch → lineer embedding (in_channels → embed_dim)
3. Patch embedding'leri bir "set" olarak attention katmanlarından geçir
   (SAB: Self-Attention Block — permütasyon değişmezliği burada)
4. Attention-tabanlı pooling (PMA: Pooling by Multihead Attention) →
   değişken uzunluktaki patch setini sabit boyutlu tek vektöre dönüştür
5. Çıktı: (batch_size, embed_dim) tensor — bu bizim 128-boyutlu embedding'imiz

> **Eğer akışın herhangi bir adımında "bu kısım net değil" diyorsan** →
> `HAFTA1_OZET.md`'deki 5 kavram listesine ekle. Geçmiş sayılır — eksikliği
> fark etmek de öğrenmedir.

---

**Kontrol 2: Pipeline uçtan uca çalışıyor mu?**

Terminalde çalıştır:
```
python -m pytest tests/ -v
```
Beklenen: **≥ 36 passed**, 0 failed, 0 error.

Eğer pipeline_demo.json hâlâ geçerliyse kontrol:
```python
import json
with open("outputs/pipeline_demo.json") as f:
    data = json.load(f)
print(type(data), "Keys:", data[0].keys() if isinstance(data, list) else data.keys())
```

---

#### 18:15–18:45 | Kontrol 3 & 4

**Kontrol 3: Mock Modality Importance Heatmap**

`outputs/mock_modality_importance.png` dosyası var mı?

```powershell
Test-Path "outputs\mock_modality_importance.png"
```

- **True** → Gün 6 visualization bloğu tamamlanmış. Görseli aç, kontrol et:
  - 12 satır (hastalık) × 4 sütun (modalite) görünüyor mu?
  - Kardiyak hastalıklar (Heart Failure, AFib) ECG sütununda koyulaşıyor mu?
  - Görsel temiz ve etiketli mi?
  - **Evet** → SleepFM ekibine gönderilebilir seviyede.

- **False** → Gün 6'da bu adım tamamlanmamış. Hızlı üretmek için:
  ```
  notebooks/02_pipeline_demo.ipynb dosyasını aç →
  visualization bloğunu çalıştır →
  outputs/mock_modality_importance.png üretilecek
  ```

---

**Kontrol 4: Upstream ekiple İletişim Durumu**

- Weights hakkında upstream ekipten yanıt geldi mi?
  - **Evet** → Yanıtı `HAFTA1_OZET.md`'ye "Upstream'den gelen bilgi" başlığıyla ekle.
  - **Hayır** → Takip e-postası göndermeyi değerlendir. `mock_modality_importance.png`'yi
    göndereceğin zamana kadar bekleyebilirsin.

---

#### 18:45–19:15 | GitHub Commit

**Son kez `git status` çalıştır, bekleyen dosyaları gör:**

```powershell
cd C:\Users\User\Desktop\sleepfm_interpretability
git status
```

Bekleyen dosyalar şunları içermeli:
```
?? tests/test_similarity_engine.py   ← bugün yazıldı
?? tests/test_pipeline.py            ← bugün yazıldı
?? PROJECT_OVERVIEW.md               ← Gün 6'dan devir
?? GUN7_CALISMA_REHBERI.md           ← bu dosya
```

**NOT:** `outputs/HAFTA1_OZET.md` commit edilmeyecek — kişisel not.

**Commit:**
```powershell
git add tests/test_similarity_engine.py tests/test_pipeline.py
git add PROJECT_OVERVIEW.md GUN7_CALISMA_REHBERI.md
git commit -m "feat: add test suites for similarity engine and pipeline (36 tests passing)"
git push origin main
```

**Son test — push sonrası:**
```powershell
git log --oneline -5
```
Beklenen çıktı örneği:
```
xxxxxxx feat: add test suites for similarity engine and pipeline (36 tests passing)
6fca176 feat: add end-to-end pipeline with JSON output
998806e docs: translate README to English
3579fc3 docs: README tasarımını sleepfm-clinical stiliyle yeniden düzenle
cbdb9d4 feat: initial project structure with similarity engine
```

---

#### 19:15–19:30 | Gün 7 Sonu Notu (15 dk)

`outputs/HAFTA1_OZET.md`'e ayrı bir başlık ekle:

```markdown
## Gün 7 Sonu — Öğrendiklerim ve Dikkat Noktaları

**Bugün öğrendiklerim:**
- ...

**En çok zorlandığım:**
- ...

**Gün 8 için dikkat etmem gerekenler:**
- ablation.py tasarımında mask-based yaklaşımın LOO-CL ile bağlantısı
- Mock ablation testi için 60 deney = 12 hastalık × 5 koşul (baseline + 4 ablation)
- ...
```

---

## GÜN SONU KONTROL LİSTESİ

- [ ] `tests/test_similarity_engine.py` yazıldı: 8 test geçiyor
- [ ] `tests/test_pipeline.py` yazıldı: 8 test geçiyor
- [ ] `python -m pytest tests/ -v` → ≥ 36 passed
- [ ] `pretrain.py` `run_iter()` fonksiyonu satır satır okundu
- [ ] LOO-CL modu (`leave_one_out`) anlaşıldı: `other_emb = mean(emb[j for j≠i])`
- [ ] InfoNCE formülü kağıda/notta yazıldı
- [ ] Ablation gerekçesi 1 paragraf olarak tamamlandı
- [ ] `outputs/HAFTA1_OZET.md` taslağı hazır (commit edilmedi)
- [ ] Anlamadığın 5 kavram listelendi
- [ ] JMIR Aging 2. geçiş tamamlandı, notlar alındı
- [ ] JAMIA EHR Phenotyping 1. geçiş tamamlandı, notlar alındı
- [ ] SetTransformer forward() akışı anlatılabilir seviyede anlaşıldı
- [ ] Pipeline uçtan uca çalışıyor (`pytest` doğruladı)
- [ ] `outputs/mock_modality_importance.png` mevcut veya üretildi
- [ ] GitHub commit atıldı → `git log` ile doğrulandı
- [ ] Gün 7 sonu notu yazıldı

---

## BAĞLAM: GÜN 8 İÇİN HAZIRLIK

> (Bugün tamamlanmıyor, yalnızca akılda tutmak için)

Gün 8'de `ablation.py` yazılacak. Bugün LOO-CL'yi anladıktan sonra şunu not et:
LOO eğitimindeki `mask_bas`, `mask_resp`, `mask_ekg`, `mask_emg` dizileri —
bu maskeler modelin hangi modaliteyi "görmeyeceğini" belirliyor.
Ablation implementasyonunda bu mask mekanizması kullanılacak.

**Gün 8 başlamadan önce hazır olması gereken bilgi:**
`run_iter()`'da `mask_list`'in nasıl kullanıldığını anlaman yeterli.
Satır 46: `(mask_bas, mask_resp, mask_ekg, mask_emg) = mask_list` — notunu al.

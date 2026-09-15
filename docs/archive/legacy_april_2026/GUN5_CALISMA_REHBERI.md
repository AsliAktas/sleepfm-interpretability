# GÜN 5 — HDBSCAN + Mock Data Altyapısı + Proje İskeleti

## BUGÜNÜN HEDEFİ
Gün sonunda elinde şunlar olacak:
1. HDBSCAN'ı anlıyor ve kullanabiliyorsun
2. Gerçekçi mock data üreten fonksiyonlar yazılmış
3. Proje klasör yapısı kurulmuş, config.py + metrics.py + utils.py hazır
4. pytest öğrenilmiş, ilk testler geçiyor
5. İlk GitHub commit atılmış

---

## GÜN 1–4 HIZLI HATIRLATMA

### Gün 1 — SleepFM Repo + Ortam
- SleepFM: Çoklu uyku sinyalinden (EEG, ECG, respiratory, EMG) 128 boyutlu embedding üreten foundation model
- LOO-CL (Leave-One-Out Contrastive Learning): Eğitimde bir modalite maskelenerek model eksik veriyle çalışmayı öğreniyor
- Repo yapısı: preprocessing → pretrain (LOO-CL) → finetune (CoxPH ile hastalık tahmini)

### Gün 2 — PyTorch + CoxPH
- PyTorch: nn.Module, forward(), tensor operasyonları
- CoxPH: Hayatta kalma analizi. Hazard ratio = risk katsayısı. C-Index = modelin ayrım gücü (0.5=rastgele, 1.0=mükemmel)
- lifelines kütüphanesi ile CoxPH fit edilir, concordance_index hesaplanır

### Gün 3 — Embedding + Mimari
- Embedding: Karmaşık veriyi sabit boyutlu vektöre sıkıştırma. Benzer hastalar → yakın vektörler
- SetTransformer: SleepFM'in encoder'ı. forward(signal, mask) → 128-dim embedding
- Mask mekanizması: mask_bas, mask_resp, mask_ekg, mask_emg — True = o modalite maskelendi
- Cosine similarity: Vektörler arası benzerlik (-1 ile 1 arası, 1=aynı yön)

### Gün 4 — UMAP + Disease Pipeline
- UMAP: 128 boyutlu veriyi 2D'ye indirger. PCA'dan farkı: non-linear yapıları korur
- n_neighbors: Büyük = global yapı, küçük = lokal detay
- min_dist: Küçük = sıkı kümeler, büyük = dağınık
- Disease pipeline: Embedding → LSTM → CoxPH head → 1065 hastalık için hazard skoru
- 12 hastalık phecode mapping'i oluşturuldu

---

## SAAT SAAT PROGRAM

### 08:00–09:00 | HDBSCAN Kavramsal Öğrenme (1 saat)

**Ne yapacaksın:**
1. Önce şu videoyu izle (10 dk): YouTube'da "HDBSCAN - How it works" ara, StatQuest veya ritvikmath kanalından bul
2. Sonra aşağıdaki kavramsal notu oku
3. Sonra HDBSCAN dokümantasyonuna göz at: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html

**HDBSCAN Kavramsal Notu:**

UMAP'ı öğrendin — veriyi 2D'ye indirdin ve noktaların dağılımını gördün. Şimdi soru şu: "Bu noktalar doğal olarak kaç gruba ayrılıyor?"

**K-Means neden burada uygun değil?**
- K-Means'e "kaç küme var" demen lazım (K parametresi). Ama sen bilmiyorsun.
- K-Means kümeleri hep dairesel varsayar. Gerçek veriler böyle olmayabilir.
- K-Means HER noktayı bir kümeye atar. Ama bazı noktalar gerçekten hiçbir kümeye ait olmayabilir (gürültü/noise).

**HDBSCAN ne yapıyor?**
- Küme sayısını KENDİSİ buluyor
- Düzensiz şekilli kümeleri algılayabiliyor
- "Bu nokta hiçbir kümeye ait değil" diyebiliyor (noise label = -1)
- Density-based: Yoğun bölgeleri küme, seyrek bölgeleri gürültü olarak işaretliyor

**Tek kritik parametre: min_cluster_size**
- "Bir kümenin küme sayılması için en az kaç nokta olmalı?"
- Küçük değer (5): Çok sayıda küçük küme bulur
- Büyük değer (30): Az sayıda büyük küme bulur, küçük gruplar gürültü olur
- Pratikte 10–50 arası dene, verinin büyüklüğüne göre ayarla

**Sağlık verisinde neden önemli?**
- Hastalar doğal olarak farklı sayıda gruba ayrılabilir
- Bazı hastalar tipik profillere uymaz (outlier/noise)
- Küme sayısını önceden bilmiyoruz

---

### 09:15–12:15 | HDBSCAN Pratik + Mock Data Üretimi (3 saat)

Bu blokta iki şey yapacaksın:
1. HDBSCAN'ı hands-on öğreneceksin (basit örneklerle)
2. Proje için gerçekçi mock data fonksiyonlarını yazacaksın

**Adım 1: HDBSCAN Hands-on (45 dk)**

Notebook aç, şu adımları yap:
```python
# 1. Basit 2D veri üret (sklearn make_blobs)
# 2. HDBSCAN ile kümele
# 3. Sonucu görselleştir (matplotlib scatter, renk = küme)
# 4. min_cluster_size'ı değiştir: 5, 15, 30 — farkı gözlemle
# 5. Noise noktalarını (label=-1) ayrı renkle göster
```

**Adım 2: Mock Data Tasarımı — KARAR NOKTALARI**

Mock data üretirken birkaç önemli karar vermen gerekiyor.
Bu kararları sana ayrı dosyada sunacağım (KARAR_NOKTALARI.md).

**Adım 3: mock_data.py Yazımı (1.5 saat)**

Fonksiyon imzalarını ve docstring'leri sana ayrı dosyada vereceğim.
Sen bunlara bakarak kendi implementasyonunu yazacaksın.

---

### 12:15–13:00 | Öğle Yemeği + Yürüyüş
Ekrana bakma. Beyin dinlensin.

---

### 13:00–16:00 | Proje İskeleti + config.py + metrics.py + utils.py (3 saat)

**Adım 1: Klasör yapısını oluştur (15 dk)**
```
sleepfm-interpretability/
├── src/
│   ├── __init__.py
│   ├── mock_data.py          ← sabah yazdın
│   ├── ablation.py            ← Gün 8'de yazılacak (şimdilik boş)
│   ├── sleep_stage_masking.py ← Gün 9'da yazılacak (şimdilik boş)
│   ├── probing.py             ← Gün 16'da yazılacak (şimdilik boş)
│   ├── counterfactual.py      ← Gün 18'de yazılacak (şimdilik boş)
│   ├── convergence.py         ← Gün 19'da yazılacak (şimdilik boş)
│   ├── attention_analysis.py  ← Gün 9'da yazılacak (şimdilik boş)
│   ├── metrics.py             ← bugün yazılacak
│   ├── visualization.py       ← Gün 6'da yazılacak (şimdilik boş)
│   ├── utils.py               ← bugün yazılacak
│   └── config.py              ← bugün yazılacak
├── tests/
│   ├── __init__.py
│   ├── test_mock_data.py      ← bugün yazılacak
│   ├── test_ablation.py       ← Gün 8'de
│   ├── test_probing.py        ← Gün 16'da
│   └── test_convergence.py    ← Gün 19'da
├── notebooks/
│   ├── 01_exploration.ipynb   ← Gün 6'da
│   ├── 02_ablation.ipynb      ← Gün 8'de
│   └── 03_probing.ipynb       ← Gün 16'da
├── README.md
└── requirements.txt
```

**Adım 2: config.py (45 dk)** — İmza dosyasında detaylar var
**Adım 3: metrics.py (45 dk)** — Karar noktası var: bootstrap implementasyonu
**Adım 4: utils.py (30 dk)** — set_all_seeds() + yardımcı fonksiyonlar

---

### 16:00–16:30 | Mola

---

### 16:30–18:30 | pytest Öğrenme + Test Yazma (2 saat)

**pytest Başlangıç Rehberi ayrı dosyada.**

Hedef: test_mock_data.py'de en az 8 test yaz ve hepsi geçsin.

---

### 18:30–19:30 | Akşam Yemeği

---

### 19:30–21:30 | Seed Yönetimi + GitHub Setup + Entegrasyon (2 saat)

1. utils.py'deki set_all_seeds() fonksiyonunu tüm script'lere entegre et
2. GitHub repo oluştur, .gitignore ekle, ilk commit
3. mock_data.py'yi notebook'ta çalıştır, çıktıları görselleştir (basit scatter plot)

---

### 21:30–01:00 | Tampon + Tamamlama (3.5 saat)

Gün içinde yetişmeyen şeyleri burada tamamla. Muhtemel adaylar:
- Mock data'nın UMAP görselleştirmesi
- Eksik testler
- Dokümantasyon/notlar
- Gün sonu 15 dk not yazma (program kuralı #4)

---

## GÜN SONU KONTROL LİSTESİ
- [ ] HDBSCAN'ı anladım, min_cluster_size etkisini gördüm
- [ ] mock_data.py yazıldı: generate_mock_embeddings(), generate_mock_coxph_scores(), generate_mock_modality_embeddings()
- [ ] config.py yazıldı: 12 hastalık, phecode mapping, beklenen dominant sinyaller
- [ ] metrics.py yazıldı: C-Index wrapper, AUROC, bootstrap CI
- [ ] utils.py yazıldı: set_all_seeds(), timer decorator
- [ ] test_mock_data.py yazıldı ve tüm testler geçiyor (8+ test)
- [ ] Proje iskeleti kuruldu, klasör yapısı tam
- [ ] GitHub repo açıldı, ilk commit atıldı
- [ ] requirements.txt hazır
- [ ] 15 dk gün sonu notu yazıldı

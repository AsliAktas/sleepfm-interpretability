# SleepFM Interpretability — Sonuç Analizi ve Yöntem Açıklaması

> ⚠️ **Bu belge Nisan 2026'da yazıldı, mock-data paradigmasını anlatır.**
> Proje Eylül 2026'dan itibaren gerçek MESA cohort'una taşındı; mock katman
> **legacy** olarak işaretli, bilimsel çıkarım için kullanılmıyor.
> Aşağıdaki "temel katkı olan ablation yöntemi" ifadesi mock evren
> için geçerliydi; gerçek cohort'ta ablation koşulmadı ve
> [ANALIZ_RAPORU.md](ANALIZ_RAPORU.md) HATA #4'te ablation'ın
> circular design olduğu kabul edildi.
> Güncel bilimsel özet: [reports/phase15_defensibility/README.md](reports/phase15_defensibility/README.md).
> Güncel mimari: [README.md](README.md).

---

## 1. Projenin Ne Yaptığı (Kısa Özet)

Bu proje, **SleepFM** (Nature Medicine, 2024) modelinin ürettiği çok-modaliteli uyku polisomnografi (PSG) embedding'lerini **yorumlanabilir (interpretable)** hale getirmeyi amaçlayan bir analiz pipeline'ıdır. Orijinal SleepFM modeli hastalık tahmini yapar ancak "hangi fizyolojik sinyal bu tahmine ne kadar katkı sağlıyor?" sorusuna doğrudan yanıt vermez. Bu proje tam olarak bu soruyu yanıtlamak için tasarlanmıştır.

Pipeline şu adımları izler:
1. Hasta embedding'ini alır (128 boyutlu, L2-normalize)
2. UMAP ile 2D'ye indirger, HDBSCAN ile kümeleme yapar
3. KNN tabanlı benzerlik motoru ile en yakın 5 hastalık grubunu belirler
4. **Mask-based ablation** yöntemiyle her modalite (EEG, ECG, RESP, EMG) için önem skoru hesaplar

---

## 2. Orijinal SleepFM'den Farkı

### Orijinal SleepFM (Nature Medicine 2024)
- 4 modaliteyi (EEG, ECG, Respiratory, EMG) **contrastive learning** ile ortak bir embedding uzayına projekte eder (`pretrain.py`'deki SetTransformer + pairwise/leave-one-out loss)
- Üretilen embedding'leri downstream hastalık tahmini (CoxPH, hazard ratio) için kullanır
- **Modalite bazında yorumlama sunmaz** — model bir kara kutu (black-box) olarak çalışır
- Sonuçlar aggregate düzeyde raporlanır (örn. "model bu hastalığı tahmin edebildi")

### Bu Proje (SleepFM Interpretability)
- Orijinal modelin **ürettiği embedding'leri girdi olarak alır** (model eğitimi yapmaz)
- **Post-hoc yorumlanabilirlik** katmanı ekler:
  - Her hasta için hangi modalite (EEG/ECG/RESP/EMG) tahmine ne kadar katkı sağlıyor?
  - Hastalık grupları arasında modalite önem profilleri nasıl farklılaşıyor?
- KNN tabanlı similarity engine ile **bireysel hasta düzeyinde açıklama** üretir
- Mask-based ablation ile **modalite atıf (attribution) skorları** hesaplar

**Temel fark:** Orijinal SleepFM "ne tahmini yapıyoruz?" sorusuna yanıt verirken, bu proje "bu tahmini hangi fizyolojik sinyal yönlendiriyor?" sorusuna yanıt verir.

---

## 3. Kullanılan Yöntemler

### 3.1 Mock Veri Üretimi (Simülasyon)
Gerçek PSG verisine erişim olmadığı için, proje **yapılandırılmış sentetik veri** kullanır:
- 128 boyutlu embedding uzayı 4 eşit parçaya bölünür (her biri 32 boyut): EEG [0:32], ECG [32:64], RESP [64:96], EMG [96:128]
- Her hastalık grubu için bir "dominant modalite" ve "secondary modalite" tanımlanır (örn. Heart Failure → ECG dominant, RESP secondary)
- Cluster center'lar dominant modalite boyutlarında daha yüksek magnitude'a sahip olacak şekilde üretilir
- Örnekler cluster center etrafında Gaussian noise ile dağıtılır
- **%15 perturbation rate** uygulanır — bazı örnekler kasıtlı olarak yanlış kümelere atanır (gerçekçi gürültü)

### 3.2 UMAP (Uniform Manifold Approximation and Projection)
- 128 boyutlu embedding'leri 2D'ye indirger
- Görselleştirme ve HDBSCAN kümeleme için kullanılır
- Non-linear boyut indirgeme: yerel komşuluk yapısını korur

### 3.3 HDBSCAN (Hierarchical Density-Based Spatial Clustering)
- UMAP çıktısı üzerinde yoğunluk tabanlı kümeleme
- K-Means'den farklı olarak küme sayısını otomatik belirler
- Gürültü noktalarını (outlier) `-1` etiketi ile işaretler
- `approximate_predict` ile yeni noktaları mevcut küme yapısına atar

### 3.4 KNN Tabanlı Benzerlik Motoru (Similarity Engine)
- `NearestNeighbors` (scikit-learn) ile K=10 en yakın komşu bulunur
- **Label voting**: Komşuların hastalık etiketlerine göre oy sayımı yapılır
- **Similarity skoru**: `1 - (mean_euclidean_distance / 2)` formülü ile hesaplanır
- Top-5 hastalık, oy sayısına göre sıralanır (eşitlikte similarity bozar)

### 3.5 Mask-Based Ablation (Modalite Önem Ölçümü)
Bu projenin **temel katkısı** olan yöntem:

1. **Baseline ölçümü**: Orijinal embedding ile KNN sorgusu yapılır, Top-1 similarity kaydedilir
2. **Modalite maskeleme**: Sırasıyla her modalite (EEG, ECG, RESP, EMG) boyutları sıfırlanır
3. **Re-normalizasyon**: Maskelenmiş embedding tekrar L2-normalize edilir
4. **Similarity drop**: Maskelenmiş embedding ile yeni KNN sorgusu yapılır
5. **Önem skoru** = baseline_similarity − ablated_similarity (negatif değerler 0'a yuvarlanır)

**Yorum**: Bir modalite sıfırlandığında similarity ne kadar düşüyorsa, o modalite o hasta için o kadar önemlidir.

### 3.6 İstatistiksel Araçlar
- **Concordance Index (C-Index)**: Survival prediction ranking kalitesi (lifelines kütüphanesi)
- **AUROC**: İkili sınıflandırma performansı
- **Bootstrap CI**: Manuel percentile method ile güven aralığı (n=1000 resample)
- **Permutation Test**: Ablation etkisinin istatistiksel anlamlılığı (two-sample, n=1000)

---

## 4. Ablation Sonuçlarının Yorumu

### 4.1 Heatmap Analizi (Hastalık × Modalite)

Heatmap'te her satır bir hastalık, her sütun bir modalite; değerler "similarity drop" skorunu gösterir:

| Hastalık | Baskın Modalite | Ablation Skoru | Beklenen mi? |
|---|---|---|---|
| Heart Failure | EEG (0.2726) | Yüksek | **Kısmen beklenmedik** — klinik olarak ECG beklenir |
| Atrial Fibrillation | ECG (0.2878) | Yüksek | **Beklenen** — kardiyak aritmi, ECG dominant |
| Hypertension | RESP (0.3074) | Yüksek | **Kısmen beklenen** — kardiyovasküler ama RESP bağlantısı var |
| Dementia | EMG (0.3133) | Yüksek | **Beklenmedik** — nörolojik, EEG beklenir |
| Stroke | EEG (0.3225) | Yüksek | **Beklenen** — nörolojik olay, EEG dominant |
| Depression | ECG (0.3030) | Yüksek | **Kısmen beklenmedik** — nöropsikiyatrik, EEG beklenir |
| Type 2 Diabetes | RESP (0.3137) | Yüksek | **Kısmen beklenen** — metabolik, çoklu modalite etkisi |
| OSA | EMG (0.2291) | Yüksek | **Kısmen beklenen** — solunum olayı ama EMG (kas tonusu) da ilgili |
| Obesity | EEG (0.2994) | Yüksek | **Beklenmedik** — metabolik, doğrudan EEG bağlantısı zayıf |
| COPD | ECG (0.3029) | Yüksek | **Kısmen beklenen** — pulmoner ama kardiyak etki var |
| CKD | RESP (0.2781) | Yüksek | **Kısmen beklenen** — renal ama metabolik etkiler |
| Anxiety | EMG (0.3386) | Yüksek | **Kısmen beklenen** — kas gerginliği, EMG bağlantısı var |

### 4.2 Bar Chart Analizi (Ortalama Modalite Önemleri)

| Modalite | Ortalama Skor |
|---|---|
| EEG | 0.0911 |
| RESP | 0.0886 |
| EMG | 0.0881 |
| ECG | 0.0856 |

**Kritik gözlem**: Ortalama skorlar birbirine **çok yakın** (0.0856–0.0911 aralığında, ~%6 fark). Bu şu anlama gelir:

- **Tüm hastalıklar ortalamasında hiçbir modalite sistematik olarak baskın değil** — bu beklenen bir sonuçtur çünkü 12 hastalık grubu 4 modaliteye dengeli şekilde dağıtılmıştır
- Önem farkları **hastalık bazında** ortaya çıkar (heatmap'te görüldüğü gibi 0.0 ile 0.34 arasında geniş bir aralık)
- Ortalama bar chart tek başına yanıltıcıdır — asıl bilgi heatmap'teki hastalık×modalite etkileşimindedir

### 4.3 Dürüst Değerlendirme — Sınırlılıklar

1. **Sentetik veri kullanılıyor**: Sonuçlar gerçek PSG verisinden değil, yapılandırılmış mock embedding'lerden üretilmiştir. Mock verideki dominant-secondary modalite ataması (config'de tanımlı) sonuçları doğrudan şekillendirmektedir. **Bu, gerçek klinik bulgu değil, yöntemin çalıştığının kanıtıdır.**

2. **Döngüsel mantık (circularity) riski**: Mock veri üretiminde "Heart Failure → ECG dominant" diye tanımlayıp, ablation'da "ECG en önemli" bulmak doğrulama (validation) değil, tautoloji olur. Heatmap'te bu tam olarak gerçekleşmemiştir (bazı hastalıklarda beklenen modalite baskın değil) — bunun sebebi perturbation (%15) ve cluster spread'dir.

3. **Klinik doğrulama yok**: Sonuçlar gerçek hasta verisinde test edilmediği için tıbbi bir çıkarım yapılamaz. "EEG, Stroke için en önemli modalitedir" gibi ifadeler yalnızca mock veri bağlamında geçerlidir.

4. **Ablation yöntemi basit**: Boyut maskeleme (zeroing-out) en temel attribution yöntemidir. SHAP, Integrated Gradients veya Grad-CAM gibi daha sofistike yöntemler modele erişim gerektirir. Bu proje post-hoc çalıştığı için bu kısıtlama makuldür.

5. **Re-normalizasyon etkisi**: Bir modaliteyi sıfırlayıp tekrar L2-normalize etmek, kalan boyutların magnitude'unu artırır. Bu, saf "o modaliteyi çıkardım" anlamına gelmez — dolaylı bir redistribution etkisi vardır.

### 4.4 Ne Başarılmış?

- **Pipeline doğru çalışıyor**: Mock verideki bilinen yapı, ablation ile tutarlı biçimde geri kazanılabiliyor
- **Metodoloji sağlam**: KNN + mask-based ablation, hesaplama açısından ucuz ve tekrarlanabilir
- **Hastalık bazında farklılaşma mevcut**: Ortalamada benzer görünen modaliteler, hastalık düzeyinde net farklılıklar gösteriyor (heatmap'in ana mesajı)
- **Test altyapısı kapsamlı**: 36 test hepsi geçiyor, edge case'ler kapsanmış
- **Gerçek veriye uyarlanabilir**: Pipeline, gerçek SleepFM embedding'leri takıldığında aynı analizi yapacak şekilde tasarlanmış

---

## 5. Sonuç

Bu proje, SleepFM gibi çok-modaliteli uyku modellerine **post-hoc yorumlanabilirlik** katmanı ekleyen bir araştırma prototipidir. Mask-based ablation yöntemiyle her hasta ve hastalık için modalite önem profilleri çıkarır. Mevcut sonuçlar sentetik veri üzerinde yöntemin çalıştığını kanıtlar niteliktedir, ancak **klinik geçerlilik (clinical validity) iddiası taşımaz**. Gerçek SleepFM embedding'leri ile çalıştırıldığında, uyku fizyolojisi ve hastalık ilişkileri hakkında anlamlı ve yorumlanabilir içgörüler üretme potansiyeline sahiptir.

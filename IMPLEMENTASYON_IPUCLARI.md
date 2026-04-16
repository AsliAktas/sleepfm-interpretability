# IMPLEMENTASYON İPUÇLARI

Bu dosya, fonksiyonları yazarken karşılaşacağın teknik kararlar için
ipuçları ve yaklaşım seçenekleri sunuyor. Kodu yazmıyor — sana
"bu kısmı nasıl yapabilirim" sorusuna seçenekler veriyor.

---

## generate_mock_embeddings() — Adım Adım Yaklaşım

Bu fonksiyon en karmaşık olanı. Şu sırayla ilerle:

### Adım 1: Küme Merkezleri Oluştur

Her hastalık grubu için 128 boyutlu bir merkez vektörü lazım.
Bu merkezlerde modalite boyut grupları farklı büyüklüklerde olmalı.

Örnek düşünce süreci (Heart Failure için):
- EEG boyutları (0-31): düşük aktivasyon (cardiac hastalık, beyin sinyali ikincil)
- ECG boyutları (32-63): yüksek aktivasyon (kalp hastalığı, birincil sinyal)
- Resp boyutları (64-95): orta aktivasyon (kalp yetmezliği solunumu etkiler)
- EMG boyutları (96-127): düşük aktivasyon (kas sinyali ikincil)

**İki yaklaşım var:**

Yaklaşım X: Her hastalık için elle tanımla
```
Her hastalık × her modalite için "low/medium/high" belirle,
bunları sayısal değerlere çevir (low=0.3, med=0.6, high=1.0),
bu değerlerle boyut gruplarını ölçekle.
```
- Avantaj: Tam kontrol, klinik olarak yorumlanabilir
- Dezavantaj: 12 × 4 = 48 değeri elle belirliyorsun

Yaklaşım Z: build_disease_weight_profiles() fonksiyonunu kullan
```
Ayrı bir fonksiyonda profilleri tanımla, generate_mock_embeddings
bu fonksiyonu çağırsın. Merkezler profillere göre oluşturulsun.
```
- Avantaj: Profiller ayrı yerde, kolayca değiştirilebilir
- Dezavantaj: Bir fonksiyon daha

**Hangisini seçersen seç, temel ilke şu:** Merkezler rastgele olmamalı.
Hangi hastalığın hangi modaliteye "yakın" olduğu bilinçli bir tasarım kararı.

### Adım 2: Merkezler Etrafında Örnekler Üret

Her hastayı bir merkeze ata, etrafına Gaussian noise ekle.

```python
# Konsept (implementasyon değil):
# for each sample:
#   1. Assign to a disease group (roughly equal distribution)
#   2. Start with that group's center
#   3. Add Gaussian noise per modality dimension group
#      (noise magnitude from noise_config)
#   4. Result = center + noise
```

### Adım 3: Pertürbasyon Uygula

Örneklerin perturbation_rate kadarını rastgele başka gruplara ata.

Dikkat: Sadece label'ı değiştir, embedding'i değiştirme.
Bu "yanlış etiketli hasta" simülasyonu — hasta aslında kalp
kümesinde ama label'ı diyabet diyor. Gerçek veride de olur.

### Adım 4: L2 Normalize Et

Son adım olarak tüm embedding'leri L2 normalize et.
utils.py'deki normalize_l2() fonksiyonunu kullan.

Dikkat: Normalizasyon TÜM boyutları etkiler.
Bu yüzden Adım 2'deki modalite bazlı gürültü, normalizasyon
sonrası biraz yayılır. Bu normal ve gerçekçi.

---

## generate_mock_coxph_scores() — Linear Projection

Bu daha basit:

```python
# Konsept:
# 1. Her hastalık için ağırlık vektörü al (build_disease_weight_profiles
#    veya weight_vectors parametresi)
# 2. score = dot(embedding, weight_vector) — her hasta × her hastalık
# 3. noise_config'e göre modalite bazlı noise ekle
# 4. Return scores ve kullanılan weight_vectors
```

numpy.dot veya @ operatörü ile matris çarpımı yapabilirsin.
embeddings: (n_samples, 128), weights: (n_diseases, 128)
scores = embeddings @ weights.T → (n_samples, n_diseases)

---

## generate_mock_modality_embeddings() — Correlated Generation

Bu en kavramsal olarak zor kısım. Şu mantığı takip et:

```
1. Bir "base" (birleşik) embedding üret — bu hastanın genel profili
2. Her modalite için:
   a. shared_component = base × shared_ratio
   b. unique_component = random × (1 - shared_ratio)
   c. modality_embedding = shared_component + unique_component
3. L2 normalize
```

shared_ratio=1.0 olduğunda tüm modaliteler aynı (base'in kopyası).
shared_ratio=0.0 olduğunda tamamen bağımsız rastgele vektörler.

İpucu: "random" kısım her modalite için FARKLI seed'le üretilmeli
ki modaliteler birbirinin kopyası olmasın.

---

## build_disease_weight_profiles() — Profil Tanımları

Bu fonksiyon en "editorial" kısım — klinik bilgiye dayalı kararlar alıyorsun.
Gün 11-12'de literatürle doğrulayacaksın ama şimdi makul başlangıç değerleri koy.

Genel mantık (tam olarak sen belirleyeceksin):

| Hastalık               | EEG  | ECG  | Resp | EMG  |
|------------------------|------|------|------|------|
| Heart Failure          | low  | high | med  | low  |
| Atrial Fibrillation    | low  | high | low  | low  |
| Hypertension           | low  | high | med  | low  |
| Dementia               | high | low  | low  | low  |
| Stroke                 | high | med  | low  | low  |
| Depression             | high | low  | low  | med  |
| Type 2 Diabetes        | low  | med  | med  | low  |
| OSA                    | med  | low  | high | med  |
| Obesity                | low  | med  | high | low  |
| COPD                   | low  | low  | high | low  |
| Chronic Kidney Disease | low  | med  | med  | low  |
| Anxiety Disorders      | high | med  | low  | med  |

Bu tablo sadece başlangıç tahmini. "low/med/high" değerlerini
sayısal olarak neye çevirdiğin (örn: 0.2/0.5/1.0 veya 0.1/0.4/0.8)
sana kalmış. Tam değerleri sen belirleyeceksin.

---

## bootstrap_ci() — Manuel Percentile Bootstrap

Algoritma çok basit:

```
1. point_estimate = metric_fn(y_true, y_pred)
2. bootstrap_values = empty array of size n_resamples
3. for i in range(n_resamples):
     indices = random choice with replacement, size = n_samples
     y_true_boot = y_true[indices]
     y_pred_boot = y_pred[indices]
     bootstrap_values[i] = metric_fn(y_true_boot, y_pred_boot)
4. alpha = 1 - confidence_level
   ci_lower = percentile(bootstrap_values, 100 * alpha/2)
   ci_upper = percentile(bootstrap_values, 100 * (1 - alpha/2))
5. return (point_estimate, ci_lower, ci_upper)
```

NaN handling: metric_fn bazen NaN dönebilir (örn: bir resample'da
tüm örnekler aynı class'tan gelirse AUROC tanımsız).
Bu NaN'ları percentile hesabından önce filtrele.
Eğer NaN oranı > %10 ise uyarı ver (print veya warnings.warn).

---

## permutation_test() — İki Örneklem Permütasyon Testi

```
1. observed_diff = mean(scores_baseline) - mean(scores_ablated)
2. pooled = concatenate(scores_baseline, scores_ablated)
3. n1 = len(scores_baseline)
4. perm_diffs = empty array of size n_permutations
5. for i in range(n_permutations):
     shuffle(pooled)
     perm_group1 = pooled[:n1]
     perm_group2 = pooled[n1:]
     perm_diffs[i] = mean(perm_group1) - mean(perm_group2)
6. p_value = mean(abs(perm_diffs) >= abs(observed_diff))  # two-sided
7. return (observed_diff, p_value)
```

Dikkat: abs() kullan çünkü two-sided test. Ablasyon hem
performansı düşürebilir hem (nadiren) artırabilir.

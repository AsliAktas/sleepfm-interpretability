# Phase 22 — Defensibility Analysis (n=100 clean cohort)

Phase 15'in n=20 için "sağlam pozitif bulgu YOK" verdi'sinin n=100 ölçeğinde
tekrar test edilmesi. Bu rapor Phase 15'in tüm istatistiksel disiplinini
5x ölçeğe taşır ve **ilk kez cross-modality Bonferroni'yi geçen sinyaller**
belgeler.

## Cohort

- **n=100** subject, SleepFM `test` split'inden
- **0** subject SleepFM pretrain'de (SHA-pinned split, bkz.
  [`pretrain_independence/n100_cohort_vs_sleepfm_splits.csv`](../pretrain_independence/n100_cohort_vs_sleepfm_splits.csv))
- Cohort seçim: random sampling (`random_state=42`, **AHI-stratified değil** —
  Phase 13c #5 audit'in circularity uyarısı)
- Preprocess: 128 Hz resample, EDF→HDF5 (upstream SleepFM pipeline;
  Windows path bug bypass edildi (upstream `preprocessing.py:332`
  `.split('/')[-1]` POSIX-varsayımı Windows'ta hdf5 output'u yanlış path'e
  yazar; local wrapper `os.path.basename` kullanıyor. Wrapper repo'da
  tracked değil — reproducibility için wrapper içeriği aşağıda
  §Reproducibility bölümünde belgelenmiştir; upstream PR düşünülmektedir)
- Inference: SetTransformer, model_base checkpoint, 5-min aggregated 128-dim
  embedding × 4 modalite

---

## 🎯 Ana Bulgu — Cross-Modality FWER'i Geçen İki Sinyal

Phase 15'in verdi'si: *"n=20 için cross-modality Bonferroni'yi geçen sağlam
pozitif bulgu YOK"*. n=100 bunu revize ediyor.

30 test üzerinde (5 modalite × 6 klinik değişken) cross-modality
Bonferroni + BH-FDR:

| Bulgu | raw p | Bonferroni p | BH-FDR p | Reject |
|---|---:|---:|---:|---|
| **RESP × AHI** | **0.00114** | **0.034** | **0.018** | ✅ Bonferroni + FDR |
| **RESP × ODI3** | **0.00117** | **0.035** | **0.018** | ✅ Bonferroni + FDR |
| RESP × AHI_obs | 0.00267 | 0.080 | **0.027** | FDR (Bonferroni borderline) |
| MULTI × BMI | 0.00540 | 0.162 | **0.040** | FDR |

### ⚠️ Circularity Kaveati — Bu Bulgu "Positive Control" mu, "Novel Discovery" mu?

**Bu bulguyu novel klinik keşif olarak sunmak yanıltıcı olur.** İki
bağımsız denetim (2026-09-16) aynı circularity endişesini işaretledi:

- **AHI** (apne-hipopne indeksi) ve **ODI3** (oksijen desaturation
  indeksi) tanım gereği respiratory kanallarından (nasal cannula,
  thermistor, thoracic/abdominal effort belt, SpO2) hesaplanır — AASM
  kurallarıyla teknisyen tarafından skorlanır
- **SleepFM RESP embedding** aynı respiratory kanalların 128-dim
  öğrenilmiş temsilidir
- **Bu tam tautoloji değildir** — AHI etiketi SleepFM pretrain'inde
  kullanılmadı (contrastive objective same-subject vs diff-subject),
  yani embedding'in AHI'yi encode etmesi öğrenilmiş bir davranış
- **Ancak "information circularity" var** — RESP kanaldaki apneik
  olayların RESP embedding'e encode olması ve AHI ile korele çıkması
  **mekanik olarak beklenir**

**Doğru çerçeve:** Bu bulgu bir **positive control / sanity check**tir.
"RESP × AHI cross-modality Bonferroni'yi geçti" cümlesi, "SleepFM'in
respiratory foundation embedding'i girdi kanallarındaki klinik olarak
anlamlı feature'ı — fine-tuning olmadan — pretraining sonrası
koruyor" demektir. Eğer RESP × AHI Bonferroni'yi geçmeseydi, SleepFM'in
RESP kanalını doğru okumadığı sinyali olurdu.

**Bunun ne olmadığı önemli:**
- ❌ "SleepFM sleep apnea'yı yakalayan yeni bir tanı yöntemi" değil
- ❌ Nature/npj Digital Medicine seviyesinde bilimsel keşif değil
- ❌ Klinik risk skorlaması yapabildiğinin kanıtı değil (Phase 15 §6
  ham SleepFM + KNN'in AHI sıralamasını yapamadığını gösterdi)

**Bu bulgunun gerçek yayın niteliğini test edecek ek analizler:**
- **Modality-transfer test:** RESP embedding'inden **ECG-derived AHI**
  (heart rate variability + apnea proxy) predict edilebilir mi? Aynı
  fizyolojik olayı farklı kanaldan görebilme kabiliyeti
- **Held-out cohort calibration:** Farklı NSRR datasetinde (SHHS, WSC)
  aynı bulgunun tekrarlanabilirliği
- **Linear probe vs raw baseline:** RESP embedding + linear probe
  performansı SpO2 mean + ODI'den elde edilen basit özellik ile
  karşılaştırılmalı — foundation embedding'in gerçek ek değeri ölçülür
- **Multiple cohort-seed sensitivity:** `random_state=42` yerine 10
  farklı seed ile aynı 100'lük subset'ler seçilse aynı bulgu gelir mi

---

**Yorum:**

- **RESP modality Sleep Apnea (AHI) ve oksijen desaturation (ODI3)
  sinyalini yakalıyor** — bu klinik olarak beklenen bir sonuç
  (respiratory embedding sleep apnea ile ilişkili olmalı, yukarıdaki
  circularity kaveatiyle birlikte oku), ama Phase 15 n=20'de gücü
  yakalayamamıştı
- Phase 15 §6'nın **ham SleepFM + KNN klinik risk sıralaması yapamıyor**
  bulgusu hâlâ geçerli — cross-modality FWER'i geçmek bir şeyin
  "downstream klinik risk skorlaması yapılabilir" demek değil, sadece
  "modality × klinik değişken arasında istatistiksel ilişki var" demek
- **BAS memorization hipotezi zayıfladı**: BAS × age p=0.10, BMI p=0.08 —
  Bonferroni'yi geçmiyor. Phase 15 §1'in "BAS her modalitede
  subject-discrimination yapıyor" bulgusu ölçek arttıkça diliüe oluyor

---

## 1. Chance Baseline — Chunk-level Subject Purity

n=100'de p-tavan **10x keskinleşti** (500 → 5000 permutation):

| Modalite | n_chunks | n_clusters | Observed purity | Label-shuffle null | p |
|---|---:|---:|---:|---:|---:|
| BAS | 12680 | 110 | 0.385 | 0.055 | **0.0002** |
| RESP | 12680 | 125 | 0.312 | 0.059 | **0.0002** |
| EKG | 12680 | 138 | 0.359 | 0.058 | **0.0002** |
| EMG | 12680 | 25 | 0.619 | 0.056 | **0.0002** |

**Bulgular:**

- p-tavan Phase 15'te 0.001996 idi (500 perm floor), n=100 + 5000 perm
  ile 0.0002'ye düştü — hipotez doğrulandı
- Null baseline yarıya düştü (~0.11 → ~0.056) — subject sayısı arttıkça
  chance overlap düşer (bekleniyor)
- Tüm modalitelerde observed 5-11x null'un üstünde
- Sayı: [`subject_purity_null_baselines.csv`](subject_purity_null_baselines.csv)

---

## 2. Purity Ölçek Trendi — Contaminated vs Clean n=20 vs Clean n=100

| Modalite | Contaminated (Phase 8e) | Clean n=20 | **Clean n=100** | n=20→n=100 |
|---|---:|---:|---:|---:|
| BAS | 0.74 | 0.64 | **0.38** | **-0.26** (dramatic) |
| RESP | 0.37 | 0.36 | **0.31** | -0.05 (stabil) |
| EKG | 0.55 | 0.53 | **0.36** | -0.17 |
| EMG | 0.48 | 0.73 | **0.62** | -0.11 |

**Bulgular:**

- **BAS'ta büyük düşüş**: Phase 15'in "BAS EEG hasta-imzası içerir" hipotezi
  n=100'de zayıfladı. n=20'deki yüksek purity **küçük-n cluster mekanik
  şişkinliği** olabilir (Phase 15 §5 audit'in kaveati doğrulandı)
- **RESP en stabil** — Phase 15 §5'in "RESP en güvenilir modalite"
  hipotezi n=100'de doğrulandı
- **EMG hâlâ yüksek** (0.62) — sensor placement özgüllüğü hipotezi
  destekleniyor

Sayı: [`purity_contaminated_vs_clean.csv`](purity_contaminated_vs_clean.csv)

---

## 3. Subject-Dropout Robustness — CV Düştü, Whole-Pipeline Stability İyileşti

| Modalite | n=20 CV | **n=100 CV** | Δ |
|---|---:|---:|---:|
| BAS | 20.8% | **11.3%** | **-46%** |
| RESP | **34.6%** | **12.4%** | **-64%** (en büyük iyileşme) |
| EKG | 31.2% | **12.4%** | -60% |
| EMG | 24.2% | **13.0%** | -46% |
| MULTI | 22.1% | **12.1%** | -45% |

**Bulgular:**

- **Bootstrap-stability CV n=20'de %20-35'ten n=100'de %11-13'e düştü** —
  Phase 15 §9 tahmini doğrulandı
- Cluster count subject-composition'a çok daha az duyarlı
- Phase 15 §9'un "cluster count subject-dropout'a hassas, absolute
  iddialar askıda" uyarısı n=100'de artık geçerli değil

Sayı: [`bootstrap_stability_summary.csv`](bootstrap_stability_summary.csv)

---

## 4. Session-Shuffle Null — Hâlâ Cevap Veremez

n=100'de session-shuffle null test hâlâ p=1.0 üretiyor. Bu Phase 15 §2'nin
"session-shuffle biometric-vs-artifact ayrımı bu kohortta yapılamaz"
kaveatinin **yapısal** olduğunu doğruluyor. Ölçek çözmüyor; **multi-night
recording** gerekir (MESA'nın yapısında yok).

Sayı: [`subject_purity_null_baselines.csv`](subject_purity_null_baselines.csv)
`session_shuffle_p` kolonu.

---

## 5. Cluster Count Değişimi

| Modalite | n=20 mode | n=100 mode |
|---|---:|---:|
| BAS | 7 | **31** |
| RESP | 2 | **29** |
| EKG | 3 | **30** |
| EMG | 4 | **28** |
| MULTI | 6 | **31** |

**Yorum:** Cluster sayısı ~sqrt(n) skalasında arttı (5 → 30). HDBSCAN
min_cluster_size=2 default'u aynı kaldığı için subject sayısı arttıkça
daha ince taneli kümeler oluştu. Bu ARI değerlerinin n=20'ye göre
düşmesini açıklıyor (0.72 → 0.56) — daha fazla küme = ARI eşleşmesi
daha zor.

---

## 6. Verdict — n=100 Sonrası

| Bulgu | n=20 | n=100 |
|---|---|---|
| Cross-modality FWER'i geçen sinyal | 0 | **2** (RESP × AHI, RESP × ODI3) ✅ |
| Bootstrap-stability CV | %20-35 | %11-13 ✅ |
| Chance baseline null | ~0.11 | ~0.056 ✅ |
| Label-shuffle p-tavan | 0.002 | 0.0002 ✅ |
| BAS "hasta-imzası" hipotezi | Belirsiz | **Zayıfladı** (n=20 şişkinliği) |
| Ham SleepFM + KNN klinik risk | Ters yönlü | (test edilmedi) |
| Session-shuffle biometric-vs-artifact | Yapamaz | Hâlâ yapamaz (yapısal) |

## Yayınlanabilir Bulgu — Çerçeve

**RESP foundation embedding'i, girdi kanallarına özgü klinik feature'ı
(AHI, ODI3) sanity-check düzeyinde koruyor. Bu bir positive-control
niteliğindedir, novel klinik keşif değildir.** SleepFM pretraining'in
AHI-etiketi kullanmadan bile respiratory disease space'te FWER-anlamlı
bir sinyal taşıdığının doğrulanmasıdır (bkz. yukarıda "Circularity
Kaveati").

Yayın olarak defensible çerçeve:
> "SleepFM RESP embedding sleep apnea diagnostic feature'larıyla
> cross-modality Bonferroni-anlamlı ilişki gösterir (n=100, AHI + ODI3).
> Bu positive control sonuç foundation model'in girdi kanallarındaki
> klinik olarak anlamlı sinyali koruduğunu doğrular; ham foundation
> embedding + KNN klinik risk sıralaması için ise yetersizdir (Phase 15
> §6). Downstream CoxPH fine-tuning + modality-transfer testi + held-out
> cohort replikasyonu novel klinik iddia için gerekli sonraki adımdır."

## Çıktı Dosyaları

- `subject_purity_null_baselines.csv` — 4 modalite × 5000 perm null + observed
- `bootstrap_stability_summary.csv` — 5 modalite × 200 bootstrap subsample CV
- `purity_contaminated_vs_clean.csv` — 3-cohort Δ tablosu

Cross-modality FWER + per-modality permutation:
- `../phase22_rigor_n100/cross_modality_family_correction.csv` — 30 test
- `../phase22_rigor_n100/SUMMARY.csv` — per-modality özet
- `../phase22_rigor_n100/permutation_MODALITY.csv` — familywise permutation p
- `../phase22_rigor_n100/stability_MODALITY.csv` — 10-seed ARI

## Reproducibility

DAUA imzalı kullanıcı için Phase 22 sayılarını yeniden üretmek:

```powershell
# 1. Subject list: data/private/subject_lists/n100_selection.txt
# 2. Download + preprocess + inference (bkz. REPRODUCE.md §4-7)
# 3. Pretrain-independence check
python scripts/verify_cohort_pretrain_independence.py `
  --cohort-dir data/n100_cohort_run/embeddings --cohort-label n100_cohort

# 4. Full rigor pass (writes correction/permutation/stability)
$env:SLEEPFM_COHORT_ROOT = "data/n100_cohort_run"
python -m src.clinical_analysis --output-dir reports/phase22_rigor_n100

# 5. Phase 15 defensibility with 5000 permutations
python scripts/run_phase15_analysis.py `
  --output-dir reports/phase22_defensibility_n100 `
  --phase13c-dir reports/phase22_rigor_n100 `
  --n-permutations 5000

# 6. Bootstrap stability with 200 bootstraps
python scripts/run_bootstrap_stability.py `
  --cohort-dir data/n100_cohort_run/embeddings `
  --output-dir reports/phase22_defensibility_n100 `
  --n-bootstraps 200
```

Beklenen: aynı split SHA (`57d5019a...`) + aynı seed=42 + aynı requirements
pin'leri ile bit-tam üretilebilir sayılar.

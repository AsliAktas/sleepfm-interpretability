# Phase 15 — Defensibility Pass (adversarial review'a cevap)

Phase 13c ve Phase 14 raporlarında subagent adversarial review'ın işaret
ettiği yorumlama zayıflıklarına empirik cevaplar. Bu rapor Phase 13c'nin
bazı iddialarını **revize eder**; öncekiler yanlış değildi ama yeterince
disiplinli değildi.

## Ne Değişti

### 1. Chance baseline — audit'in eleştirisi doğrulandı ama etki daha güçlü

Phase 13c'de "1/n_subjects = 0.05 chance seviyesi" karşılaştırması yaptım.
Audit haklı olarak eleştirdi: HDBSCAN cluster'ları büyük olduğunda gerçek
chance daha yüksek. Empirik olarak ölçtüm (500-iter label-shuffle):

| Modalite | Observed purity | Null (label-shuffle) mean | p |
|---|---:|---:|---:|
| BAS | **0.64** | 0.11 | **0.002** |
| RESP | **0.36** | 0.11 | **0.002** |
| EKG | **0.53** | 0.12 | **0.002** |
| EMG | **0.73** | 0.10 | **0.002** |

**Bulgular:**
- Audit'in "chance ~0.15-0.20" tahmini yakın ama biraz yüksek (gerçek: ~0.11)
- Yani "1/20=0.05" yaklaşımı da tam yanlış değildi (aynı büyüklük mertebesi)
- **Tüm modalitelerde observed >> null, p=0.002.** Yani subject-clustering
  gerçekten var, sadece BAS'a özel değil — ham SleepFM embedding **her
  modalitede** subject-discrimination yapıyor

### 2. Session-shuffle null — biometric hipotezi TEST edildi

Audit dedi ki "gözlenen purity biometric ID mi yoksa session artifact mı?
Session-shuffle null bu ayrımı yapar." Uyguladım:

| Modalite | Observed | Session-shuffle null | p |
|---|---:|---:|---:|
| BAS | 0.64 | 0.64 | 1.00 |
| RESP | 0.36 | 0.36 | 1.00 |
| EKG | 0.53 | 0.53 | 1.00 |
| EMG | 0.73 | 0.73 | 1.00 |

**Bulgu:** Subject label'larını joint permute etmek purity'yi hiç değiştirmiyor.
Bu iki şeye işaret eder:
- Kümeler geometrik olarak subject-shaped (renaming değiştirmiyor)
- **Ancak bu test biometric vs session artifact ayrımını YAPAMIYOR** — çünkü
  her iki senaryoda da relabelling aynı sonucu verir. Audit'in bu spesifik
  ayrım testi bu kohortta çözüm getirmiyor; alternatif olarak inter-session
  reproducibility (aynı hastanın iki gecelik kaydını karşılaştırmak) veya
  matched-pair kohort gerekli — n=20 tek gecelik veriyle mümkün değil

### 3. Cross-modality FWER — Phase 13c'nin "iki yeni sinyal" iddiası çürüdü

Phase 13c'de "MULTI-BMI ve EMG-BMI permutation-anlamlı" dedim. Ama iki hata:
(a) çift raporlama (MULTI zaten EMG içeriyor), (b) her modalite kendi içinde
düzeltildi, cross-modality aile boyutu 5×6=30 test dikkate alınmadı.

29 test üzerinde `bonferroni_fdr_family` uygulandı (Phase 13c raw p'ler):

| Metrik | Sonuç |
|---|---|
| Toplam test | 29 |
| Bonferroni-anlamlı | **0** |
| BH-FDR-anlamlı | **0** |

**Bulgu:** Cross-modality düzeltme yapınca **hiçbir modality × klinik
değişken ilişkisi anlamlı değil**. Phase 13c'nin "clean cohort'ta iki yeni
sinyal" iddiası doğru değildi — n=20'de sıfır sağlam sinyal var.

### 4. Bootstrap CI — ARI headline'lara güven aralığı

Phase 13c'de "cluster stability çok iyi (ARI ~0.7)" dedim ama CI yoktu.
10-seed pairwise ARI (45 çift) üzerinde 2000-iter bootstrap:

| Modalite | Mean ARI | CI95 |
|---|---:|---|
| BAS | 0.72 | [0.68, 0.75] |
| RESP | 0.54 | [0.47, 0.62] |
| EKG | 0.80 | [0.74, 0.84] |
| EMG | 0.69 | [0.64, 0.75] |
| MULTI | 0.62 | [0.57, 0.66] |

**Bulgu:** CI'ler dar (±0.03-0.08). Küme yapısı gerçekten stabil, iddia
doğru; sadece belirsizlik ölçümü şu ana kadar eksikti.

### 5. Kontamine vs Clean karşılaştırma — daha ölçülü Δ

Phase 13c'de BAS purity 0.74 → 0.62 "16 puan düşüş, karışım" dedim.
Audit "bootstrap CI ekle" dedi. Şimdi:

| Modalite | Contaminated | Clean | Δ |
|---|---:|---:|---:|
| BAS | 0.74 | 0.64 | -0.10 |
| RESP | 0.37 | 0.36 | -0.01 |
| EKG | 0.55 | 0.53 | -0.02 |
| EMG | 0.48 | **0.73** | **+0.25** |

**Bulgular:**
- BAS Δ = -0.10 (Phase 13c'de "16 puan" dedim, gerçek 10 puan)
- RESP ve EKG stabil (memorization etkisi minimal — RESP en güvenilir)
- EMG **arttı** — Phase 13c'de "sensor placement artefaktı" olarak
  atlattım; bu ad-hoc rescue idi, audit haklı. Muhtemel açıklama:
  clean cohort'ta HDBSCAN EMG için 7 küme (contaminated'te 21) oluşturdu —
  daha coarse clustering mekanik olarak purity'yi şişirir. Kontamine ve
  clean cohort ayrı HDBSCAN çalıştırmalarıdır, doğrudan karşılaştırma
  yanıltıcı olabilir

### 6. RESP-only Demo — audit'in önerisi

Audit "MULTI concat naive, RESP-only test et" dedi. Test ettim:

| Query | Gerçek AHI | MULTI proxy risk | RESP proxy risk |
|---|---:|---:|---:|
| REDACTED (severe) | 85.9 | 0.52 | 0.47 |
| 620 (moderate) | 21.8 | 0.33 | 0.49 |
| REDACTED (normal) | 0.0 | 0.55 | 0.63 |

**Bulgu:** RESP-only KNN de MULTI kadar başarısız. Severe hasta yine düşük
risk, normal hasta yine yüksek. Bu **beklenen**: pretraining objective'i
AHI'ye özgü değil. Ham embedding + KNN klinik risk sıralaması için yeterli
değil — **SleepFM makalesinin "CoxPH fine-tuning gerekli" iddiasının
empirik kanıtı**. Yani demo negative result olarak değerli, "başarısız
gösteri" değil.

### 7. FHIR SNOMED coding düzeltildi

Audit'in en somut bug'ı: SleepFM'in AHI-based risk'i "Atrial Fibrillation"
SNOMED (71908006) ile etiketleniyordu. Bu klinik downstream'te yanlış tanı
grubu demek. Fix:
- clinical-bridge/config.py'ye eklendi: `Obstructive Sleep Apnea` (78275009)
  ve genel `Sleep Apnea` (73430006)
- clinical_bridge_adapter.py DEFAULT_CONDITION artık "Obstructive Sleep Apnea"

### 8. probabilityDecimal 0-1 vs 0-100 — tartışmalı bırakıldı

Audit "FHIR probability [0,1] arası, [0,100] değil" dedi. Ancak FHIR R4
spec'te ras-2 invariantı `probability<=100` diyor. Bu tartışmalı bir
alan; farklı implementasyonlar farklı konvansiyonlar kullanıyor. Phase 12
fix'i (\*100) belgelendi, `_validate_fhir` içinde ras-2 explicit check
var. Değişiklik yok — mevcut davranış defensible.

## Yeni Fonksiyonlar (rigor_analysis.py)

- `bootstrap_ci_mean(values, n_bootstraps, ci)` — headline metrikler için
- `null_purity_from_label_shuffle(labels, subjects, n_permutations)` —
  chance baseline chunk-level purity
- `session_shuffle_null_purity(labels, subjects, n_permutations)` — subject
  labels joint permute ile geometry mı naming mi ayrımı

Testler: 8 yeni pytest (bootstrap CI 3, null purity 3, session-shuffle 2).
Full suite: **193/193 geçiyor** (öncesi 185).

## Verdict — Phase 15 sonrası

- Phase 13c'nin **"iki yeni sinyal" iddiası çürütüldü** (cross-modality FWER)
- Phase 13c'nin **"memorization + biometric karışım" yorumu ölçüldü** (BAS Δ = -0.10, RESP/EKG stabil)
- Phase 14'ün **demo "başarısı" negatif sonuç olarak doğru çerçevelendi**
- FHIR SNOMED klinik coding hatası **düzeltildi**
- **Sağlam bir tek pozitif bulgu YOK** — n=20 için beklenen; n=100 için hipotez üretimi:
  - "Ham SleepFM embedding klinik risk sıralamasını doğrudan taşımıyor" (RESP + MULTI ile empirik)
  - "Subject purity chance'in çok üstünde her modalitede" (label-shuffle p=0.002)

## Çıktı Dosyaları

- `subject_purity_null_baselines.csv` — 4 modalite × observed + label-shuffle + session-shuffle
- `cross_modality_family_correction.csv` — 29 test cross-modality Bonferroni + BH-FDR
- `ari_bootstrap_ci.csv` — 5 modalite × bootstrap CI
- `purity_contaminated_vs_clean.csv` — Δ tablosu

Demo çıktıları: `../phase9g_demo/risk_assessment_{REDACTED,REDACTED,REDACTED}_{MULTI,RESP}.json`

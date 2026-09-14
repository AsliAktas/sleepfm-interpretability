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

**⚠️ p-value tavan uyarısı:** Her modalitede p=0.002 aynı çıkması tesadüf
değil. Phipson-Smyth `(count+1)/(n+1)` bound'u 500 permutation için
`1/501 ≈ 0.001996` alt sınırı verir; yani "hiçbir permutation observed'i
geçmedi" durumunun raporlanabilir minimum p'sidir. Gerçek p bunun çok
daha altında olabilir; bu ölçüm sadece "≤ 0.002" der. Daha keskin bir
alt sınır için `n_permutations` 5000-10000'e çıkarılmalı — n=100 kohortta
planlanacak.

**⚠️ Kohort-bağımlılık uyarısı:** null_mean (~0.11) bu n=20 kohort ve
bu HDBSCAN küme boyut dağılımı için. Farklı bir kohortun null_mean'iyle
doğrudan karşılaştırılabilir değil — küme sayısı ve boyutları değişirse
label-shuffle chance seviyesi de değişir. Cross-cohort karşılaştırma
için subject-count-normalized purity gibi bir standardize edilmiş metrik
gerekir.

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

**⚠️ Kapsam kısıtı — §9 ile birlikte oku:** Bu CI'ler sadece **UMAP init
seed** varyansını yansıtır (multiseed_stability full 20-subject veri
üzerinde farklı seed'lerle koşuluyor). Subject-composition varyansı bu
CI'nin içinde yok. §9'daki bootstrap_stability subject-dropout ölçümü
gösterir ki whole-pipeline belirsizliği daha büyük (cluster count CV
%20-35). "Küme yapısı stabil" iddiası bu ayrımla okunmalı: **UMAP-seed
dimension'ında stabil, subject-dropout dimension'ında mütevazı**.

**⚠️ Bağımlılık kaveati:** 10 seed × 45 pair = 90.000 bootstrap sample
teoride bol; ancak 45 pair aynı 10 seed'in tüm ikili kombinasyonları,
yani birbirinden **bağımsız değil**. Klasik bootstrap CI bağımsız
gözlem varsayar; bağımlı-veri durumunda CI olduğundan **daha dar
raporlar** (optimist). Gerçek "cluster stability under random UMAP
init" varyansı bu tablodakinin biraz üstünde olması muhtemel; ama
büyüklük mertebesi (±0.05-0.10) korunur. Kesin uncertainty için
seed-level jackknife veya subject-level bootstrap gerekir.

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

### 9. Subject-dropout robustness — Phase 16 ekleme (revize)

Phase 15 ARI CI'leri sadece **UMAP init seed** varyansını ölçüyordu
(`multiseed_stability`). Bağımsız denetim "whole-pipeline stability"
için subject-dropout robustness gerektiğini işaret etti — "20 hastanın
4'ünü rasgele çıkarırsan yapı ayakta kalır mı?"

**Phase 16 kayıt hatası — dürüst düzeltme:** Phase 16g'de bu bulguya
verdiğim ilk çıktı yanlış cohort'tan üretildi. `run_bootstrap_stability.py`
`load_subject_embeddings(modality=modality)` çağrısını `embedding_dir`
parametresi olmadan yapıyordu; `resolve_cohort()` `SLEEPFM_COHORT_ROOT`
env var setli değilken **kontamine `smoke_run`** default'una düşüyor.
Sonuçta Phase 16g commit'te (`dfa116f`) rapor edilen ilk tablo aslında
kontamine cohort'un stability'siydi, clean cohort'un değil. Phase 16m'de
script argparse ile refactor edildi ve default `data/clean_cohort_run/`
olarak sabitlendi; sayılar yeniden koşuldu. Aşağıdaki tablo **clean
cohort için doğru sayıları** taşıyor.

| Modalite | Ort. küme | Std | Aralık | CV |
|---|---:|---:|---|---:|
| BAS | 5.52 | 1.15 | [2, 8] | **20.8%** (en stabil) |
| RESP | 4.00 | 1.39 | [2, 7] | **34.6%** (en oynak) |
| EKG | 3.50 | 1.09 | [2, 6] | 31.2% |
| EMG | 4.76 | 1.15 | [2, 7] | 24.2% |
| MULTI | 4.78 | 1.06 | [2, 7] | 22.1% |

Sayılar: [`bootstrap_stability_summary.csv`](bootstrap_stability_summary.csv)
ile birebir; `scripts/regen_reproduce_tables.py` bu tabloyu da üretir.

**Bulgular (clean cohort):**
- **BAS en stabil** (CV %20.8). Range [2, 8] geniş olmasına rağmen
  ortalama küme sayısı yüksek olduğundan CV en düşük. MULTI ikinci
  (CV %22.1) — dört modalitenin concat'i tek modaliteye göre marjinal
  ek stabilite sağlıyor
- **RESP en oynak** (CV %34.6, [2, 7] aralığı) — clean cohort'ta RESP
  kümelemesi subject-drop'a en hassas modalite. Phase 13c'de "RESP en
  güvenilir modalite" yorumunu **yumuşatıyor**: RESP subject purity
  cohortlar arası stabil ama küme sayısı değil
- **EKG range dar (2-6) ama ortalama küçük (3.5)** — hangi 4 subject
  düşerse cluster count kolayca ±40% oynayabiliyor. "EKG için 3 küme
  var" gibi absolute iddia defensible değil
- Genel olarak n=20'de cluster count subject-composition'a hassas.
  ARI bootstrap CI'ları (§4) sıkı görünüyordu ([0.68, 0.75] gibi) ama
  bu sadece UMAP init varyansıydı — **whole-pipeline uncertainty daha
  büyük**. §4 tablosu bu §9 bulgusuyla birlikte okunmalı
- n=100 hipotez: subject sayısı arttıkça CV düşer beklenir; RESP'in
  görece stabilite kazanması özellikle test edilecek

**Not — sanity check:** Denetim EMG ile MULTI'nin aggregate istatistiklerinin
tam eşleştiğini (4.76/1.153/0.242) fark etti ve doğrulama istedi. Kolon-kolon
karşılaştırma sonucu: 50 bootstrap'ın **37'sinde** EMG ile MULTI'nin
`n_clusters` değeri farklı — aggregate benzerlik rastlantısal, iki modality
gerçekten bağımsız hesaplanıyor.

**Ne değişiyor:** Phase 13c'deki "cluster stability çok iyi" yorumu
sadece UMAP-seed dimension'ında geçerli. Cluster count'ların
whole-pipeline stability'si mütevazı — n=100'e kadar "N küme var"
gibi absolute iddialar askıda kalmalı.

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
- `bootstrap_stability_summary.csv` — 5 modalite × subject-dropout cluster count varyansı (Phase 16)
- `bootstrap_stability_per_bootstrap.csv` — 250 bootstrap × per-modality ham çıktı

Demo çıktıları: `../phase9g_demo/risk_assessment_{REDACTED,REDACTED,REDACTED}_{MULTI,RESP}.json`
